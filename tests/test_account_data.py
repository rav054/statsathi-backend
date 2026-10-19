"""Exporting everything an account holds, and deleting an account."""
import io
import json
import unittest
import uuid
import zipfile
from datetime import datetime

import pandas as pd

from tests.support import client_and_headers

from app import database
from app.models import AnalysisResult, AuthToken, Dataset, Project, ProjectMember, User

PASSWORD = "api-tests-password"
CSV = pd.DataFrame({"treatment": ["T1", "T2"], "yield": [4.2, 5.1]}).to_csv(index=False).encode()
RESULT = {"module": "anova", "title": "Yield ANOVA", "parameters": {"response": "yield"},
          "result": {"p_value": 0.012}, "report": {"sections": []}}


def new_user():
    email = f"data-{uuid.uuid4().hex[:10]}@example.com"
    client, headers = client_and_headers(email)
    with database.SessionLocal() as db:
        db.query(User).filter(User.email == email).update({User.email_verified_at: datetime.utcnow()})
        db.commit()
    return client, headers, email


class AccountDataTests(unittest.TestCase):
    def setUp(self):
        self.client, self.headers, self.email = new_user()
        self.project = self.client.post("/workspace/projects", headers=self.headers, json={
            "name": "Maize/N trial", "tasks": [{"id": 1, "text": "Run ANOVA", "completed": False}]}).json()
        pid = self.project["id"]
        self.dataset = self.client.post(f"/workspace/projects/{pid}/datasets", headers=self.headers,
                                        files={"file": ("plots.csv", CSV, "text/csv")}).json()
        self.client.post(f"/workspace/projects/{pid}/results", headers=self.headers,
                         json={**RESULT, "dataset_id": self.dataset["id"]})
        self.client.post(f"/workspace/projects/{pid}/recipes", headers=self.headers,
                         json={"name": "Drop gaps", "steps": [{"type": "drop_missing"}]})

    def share(self, owner_headers, project_id, role="editor"):
        _, headers, email = new_user()
        member = self.client.post(f"/workspace/projects/{project_id}/members", headers=owner_headers,
                                  json={"email": email, "role": role}).json()
        self.client.post(f"/workspace/invitations/{member['id']}/accept", headers=headers)
        return headers, email

    def test_export_contains_every_file_result_and_recipe(self):
        response = self.client.get("/auth/account/export", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertIn("statsathi-export-", response.headers["content-disposition"])

        archive = zipfile.ZipFile(io.BytesIO(response.content))
        names = archive.namelist()
        folder = f"projects/{self.project['id']}-Maize_N trial"
        self.assertIn("README.txt", names)
        self.assertEqual(json.loads(archive.read("account.json"))["email"], self.email)
        self.assertEqual(json.loads(archive.read(f"{folder}/project.json"))["tasks"][0]["text"], "Run ANOVA")
        self.assertEqual(archive.read(f"{folder}/datasets/{self.dataset['id']}-plots.csv"), CSV)
        [result_name] = [n for n in names if "/results/" in n]
        result = json.loads(archive.read(result_name))
        self.assertEqual((result["result"], result["report"]), ({"p_value": 0.012}, {"sections": []}))
        [recipe_name] = [n for n in names if "/recipes/" in n]
        self.assertEqual(json.loads(archive.read(recipe_name))["statsathi_recipe"], 1)

    def test_export_of_a_shared_project_holds_only_what_the_member_added(self):
        member_headers, _ = self.share(self.headers, self.project["id"])
        self.client.post(f"/workspace/projects/{self.project['id']}/results", headers=member_headers,
                         json={**RESULT, "title": "Member result"})
        archive = zipfile.ZipFile(io.BytesIO(self.client.get("/auth/account/export", headers=member_headers).content))
        names = archive.namelist()
        self.assertFalse(any("/datasets/" in n for n in names))
        [result_name] = [n for n in names if "/results/" in n]
        self.assertIn("Member result", result_name)
        account = json.loads(archive.read("account.json"))
        self.assertEqual(account["projects_shared_with_me"][0]["role"], "editor")

    def test_deleting_needs_the_password_and_the_word_delete(self):
        wrong = self.client.post("/auth/account/delete", headers=self.headers, json={"password": "nope", "confirmation": "DELETE"})
        self.assertEqual(wrong.status_code, 400)
        unconfirmed = self.client.post("/auth/account/delete", headers=self.headers, json={"password": PASSWORD, "confirmation": "yes"})
        self.assertEqual(unconfirmed.status_code, 400)
        self.assertEqual(self.client.get("/auth/me", headers=self.headers).status_code, 200)

    def test_deleting_removes_the_account_and_its_projects_but_keeps_contributions_elsewhere(self):
        other_client, other_headers, other_email = new_user()
        others_project = other_client.post("/workspace/projects", headers=other_headers, json={"name": "Other lab"}).json()
        member = other_client.post(f"/workspace/projects/{others_project['id']}/members", headers=other_headers,
                                   json={"email": self.email, "role": "editor"}).json()
        self.client.post(f"/workspace/invitations/{member['id']}/accept", headers=self.headers)
        contribution = self.client.post(f"/workspace/projects/{others_project['id']}/results", headers=self.headers, json=RESULT).json()
        viewer_headers, _ = self.share(self.headers, self.project["id"], "viewer")

        response = self.client.post("/auth/account/delete", headers=self.headers, json={"password": PASSWORD, "confirmation": "DELETE"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((response.json()["projects"], response.json()["reassigned_items"]), (1, 1))

        self.assertEqual(self.client.get("/auth/me", headers=self.headers).status_code, 401)
        self.assertEqual(self.client.post("/auth/login", data={"username": self.email, "password": PASSWORD}).status_code, 401)
        self.assertEqual(self.client.get(f"/workspace/projects/{self.project['id']}", headers=viewer_headers).status_code, 404)

        kept = other_client.get(f"/workspace/projects/{others_project['id']}", headers=other_headers).json()
        self.assertEqual([(r["id"], r["added_by"]["email"]) for r in kept["results"]], [(contribution["id"], other_email)])
        self.assertEqual(kept["members"], [])

        with database.SessionLocal() as db:
            self.assertIsNone(db.query(User).filter(User.email == self.email).first())
            self.assertEqual(db.query(Project).filter(Project.id == self.project["id"]).count(), 0)
            self.assertEqual(db.query(Dataset).filter(Dataset.project_id == self.project["id"]).count(), 0)
            self.assertEqual(db.query(AnalysisResult).filter(AnalysisResult.project_id == self.project["id"]).count(), 0)
            self.assertEqual(db.query(ProjectMember).filter(ProjectMember.project_id == self.project["id"]).count(), 0)
            self.assertEqual(db.query(ProjectMember).filter(ProjectMember.email == self.email).count(), 0)
            self.assertEqual(db.query(AuthToken).join(User, User.id == AuthToken.user_id, isouter=True)
                             .filter(User.id.is_(None)).count(), 0)

        # The address is free again
        again = self.client.post("/auth/signup", json={"email": self.email, "password": PASSWORD, "full_name": "Back again"})
        self.assertEqual(again.status_code, 201)


if __name__ == "__main__":
    unittest.main()
