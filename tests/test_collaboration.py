"""Sharing projects: invitations, the viewer/editor/owner rules and leaving or removing members."""
import unittest
import uuid
from datetime import datetime

import pandas as pd

from tests.support import client_and_headers

from app import database, emailer
from app.models import ProjectMember, User


def new_user(verified=True):
    email = f"collab-{uuid.uuid4().hex[:10]}@example.com"
    client, headers = client_and_headers(email)
    if verified:
        with database.SessionLocal() as db:
            db.query(User).filter(User.email == email).update({User.email_verified_at: datetime.utcnow()})
            db.commit()
    return client, headers, email


CSV = pd.DataFrame({"variety": ["A", "B", "C"], "yield": [4.1, 3.8, 4.5]}).to_csv(index=False).encode()
RESULT = {"module": "anova", "title": "Yield ANOVA", "parameters": {"response": "yield"}, "result": {"p": 0.03}}


class CollaborationTests(unittest.TestCase):
    def setUp(self):
        self.client, self.owner, self.owner_email = new_user()
        self.project = self.client.post("/workspace/projects", headers=self.owner, json={"name": "Rice N trial"}).json()
        self.pid = self.project["id"]

    def invite(self, email, role="viewer"):
        response = self.client.post(f"/workspace/projects/{self.pid}/members", headers=self.owner,
                                    json={"email": email.upper(), "role": role})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def join(self, role="viewer"):
        _, headers, email = new_user()
        member = self.invite(email, role)
        invitations = self.client.get("/workspace/invitations", headers=headers).json()["invitations"]
        [invitation] = [i for i in invitations if i["project_id"] == self.pid]
        accepted = self.client.post(f"/workspace/invitations/{invitation['id']}/accept", headers=headers)
        self.assertEqual(accepted.status_code, 200, accepted.text)
        return headers, member

    def test_invitation_is_emailed_and_nothing_is_shared_until_accepted(self):
        _, headers, email = new_user()
        member = self.invite(email)
        self.assertEqual((member["email"], member["status"], member["email_sent"]), (email, "invited", True))
        self.assertIn("Rice N trial", emailer.OUTBOX[-1]["subject"])
        self.assertEqual(emailer.OUTBOX[-1]["to"], email)

        self.assertEqual(self.client.get(f"/workspace/projects/{self.pid}", headers=headers).status_code, 404)
        self.assertNotIn(self.pid, [p["id"] for p in self.client.get("/workspace/projects", headers=headers).json()])

        [invitation] = self.client.get("/workspace/invitations", headers=headers).json()["invitations"]
        self.assertEqual((invitation["project_name"], invitation["role"], invitation["invited_by"]["email"]),
                         ("Rice N trial", "viewer", self.owner_email))
        self.client.post(f"/workspace/invitations/{invitation['id']}/accept", headers=headers)

        listed = {p["id"]: p for p in self.client.get("/workspace/projects", headers=headers).json()}
        self.assertEqual(listed[self.pid]["role"], "viewer")
        self.assertEqual(listed[self.pid]["owner"]["email"], self.owner_email)
        owner_view = self.client.get(f"/workspace/projects/{self.pid}", headers=self.owner).json()
        self.assertEqual([(m["email"], m["status"]) for m in owner_view["members"]], [(email, "active")])

    def test_unconfirmed_addresses_see_no_details_and_cannot_accept(self):
        _, headers, email = new_user(verified=False)
        member = self.invite(email)
        body = self.client.get("/workspace/invitations", headers=headers).json()
        self.assertEqual(body, {"invitations": [], "pending_count": 1, "needs_verification": True})
        refused = self.client.post(f"/workspace/invitations/{member['id']}/accept", headers=headers)
        self.assertEqual(refused.status_code, 403)
        self.assertIn("Confirm your email", refused.json()["detail"])

    def test_someone_else_cannot_accept_or_see_an_invitation(self):
        _, _, email = new_user()
        member = self.invite(email)
        _, stranger, _ = new_user()
        self.assertEqual(self.client.post(f"/workspace/invitations/{member['id']}/accept", headers=stranger).status_code, 404)
        self.assertEqual(self.client.get("/workspace/invitations", headers=stranger).json()["invitations"], [])

    def test_viewer_can_read_everything_but_change_nothing(self):
        dataset = self.client.post(f"/workspace/projects/{self.pid}/datasets", headers=self.owner,
                                   files={"file": ("trial.csv", CSV, "text/csv")}).json()
        result = self.client.post(f"/workspace/projects/{self.pid}/results", headers=self.owner, json=RESULT).json()
        viewer, _ = self.join("viewer")

        project = self.client.get(f"/workspace/projects/{self.pid}", headers=viewer).json()
        self.assertEqual(project["role"], "viewer")
        self.assertEqual(project["datasets"][0]["added_by"]["email"], self.owner_email)
        self.assertFalse(project["datasets"][0]["can_change"])
        self.assertEqual(self.client.get(f"/workspace/datasets/{dataset['id']}/download", headers=viewer).content, CSV)
        self.assertEqual(self.client.get(f"/workspace/results/{result['id']}", headers=viewer).json()["result"], {"p": 0.03})
        self.assertEqual(self.client.get(f"/workspace/projects/{self.pid}/methods", headers=viewer).status_code, 200)
        self.assertEqual(self.client.post(f"/workspace/projects/{self.pid}/report", headers=viewer, json={}).status_code, 200)

        for response in (
            self.client.post(f"/workspace/projects/{self.pid}/results", headers=viewer, json=RESULT),
            self.client.post(f"/workspace/projects/{self.pid}/datasets", headers=viewer, files={"file": ("x.csv", CSV, "text/csv")}),
            self.client.patch(f"/workspace/projects/{self.pid}", headers=viewer, json={"status": "Completed"}),
            self.client.delete(f"/workspace/results/{result['id']}", headers=viewer),
            self.client.delete(f"/workspace/projects/{self.pid}", headers=viewer),
            self.client.post(f"/workspace/projects/{self.pid}/members", headers=viewer, json={"email": "x@example.com"}),
        ):
            self.assertEqual(response.status_code, 403, response.request.url)
        self.assertIn("Ask the owner to make you an editor", self.client.delete(f"/workspace/results/{result['id']}", headers=viewer).json()["detail"])

    def test_editor_adds_work_but_changes_only_their_own(self):
        owners_result = self.client.post(f"/workspace/projects/{self.pid}/results", headers=self.owner, json=RESULT).json()
        editor, _ = self.join("editor")

        dataset = self.client.post(f"/workspace/projects/{self.pid}/datasets", headers=editor, files={"file": ("e.csv", CSV, "text/csv")})
        self.assertEqual(dataset.status_code, 201)
        mine = self.client.post(f"/workspace/projects/{self.pid}/results", headers=editor,
                                json={**RESULT, "dataset_id": dataset.json()["id"]})
        self.assertEqual(mine.status_code, 201)
        recipe = self.client.post(f"/workspace/projects/{self.pid}/recipes", headers=editor,
                                  json={"name": "Clean", "steps": [{"type": "drop_missing"}]})
        self.assertEqual(recipe.status_code, 201, recipe.text)
        self.assertIn(recipe.json()["id"], [r["id"] for r in self.client.get("/workspace/recipes", headers=self.owner).json()])
        self.assertEqual(self.client.patch(f"/workspace/projects/{self.pid}", headers=editor, json={"status": "In Progress"}).status_code, 200)
        self.assertEqual(self.client.patch(f"/workspace/projects/{self.pid}", headers=editor, json={"name": "Renamed"}).status_code, 403)

        refused = self.client.delete(f"/workspace/results/{owners_result['id']}", headers=editor)
        self.assertEqual(refused.status_code, 403)
        self.assertIn("only what they added", refused.json()["detail"])
        self.assertEqual(self.client.delete(f"/workspace/results/{mine.json()['id']}", headers=editor).status_code, 204)
        self.assertEqual(self.client.delete(f"/workspace/results/{owners_result['id']}", headers=self.owner).status_code, 204)

    def test_owner_changes_roles_removes_members_and_their_work_stays(self):
        editor, member = self.join("editor")
        result = self.client.post(f"/workspace/projects/{self.pid}/results", headers=editor, json=RESULT).json()

        demoted = self.client.patch(f"/workspace/projects/{self.pid}/members/{member['id']}", headers=self.owner, json={"role": "viewer"})
        self.assertEqual(demoted.json()["role"], "viewer")
        self.assertEqual(self.client.post(f"/workspace/projects/{self.pid}/results", headers=editor, json=RESULT).status_code, 403)

        self.assertEqual(self.client.delete(f"/workspace/projects/{self.pid}/members/{member['id']}", headers=self.owner).status_code, 204)
        self.assertEqual(self.client.get(f"/workspace/projects/{self.pid}", headers=editor).status_code, 404)
        self.assertEqual(self.client.get(f"/workspace/results/{result['id']}", headers=editor).status_code, 404)
        kept = self.client.get(f"/workspace/projects/{self.pid}", headers=self.owner).json()["results"]
        self.assertEqual([(r["id"], r["added_by"]["email"], r["can_change"]) for r in kept], [(result["id"], self.owner_email, True)])

    def test_members_can_leave_but_not_remove_others(self):
        first, first_member = self.join("editor")
        _, second_member = self.join("viewer")
        self.assertEqual(self.client.delete(f"/workspace/projects/{self.pid}/members/{second_member['id']}", headers=first).status_code, 403)
        self.assertEqual(self.client.delete(f"/workspace/projects/{self.pid}/members/{first_member['id']}", headers=first).status_code, 204)
        self.assertEqual(self.client.get(f"/workspace/projects/{self.pid}", headers=first).status_code, 404)

    def test_invitation_rules(self):
        self.assertEqual(self.client.post(f"/workspace/projects/{self.pid}/members", headers=self.owner,
                                          json={"email": self.owner_email}).status_code, 400)
        _, _, email = new_user()
        self.invite(email)
        again = self.client.post(f"/workspace/projects/{self.pid}/members", headers=self.owner, json={"email": email})
        self.assertEqual(again.status_code, 409)
        self.assertIn("already been invited", again.json()["detail"])
        self.assertEqual(self.client.post(f"/workspace/projects/{self.pid}/members", headers=self.owner,
                                          json={"email": "someone@example.com", "role": "owner"}).status_code, 422)

        _, stranger, _ = new_user()
        self.assertEqual(self.client.post(f"/workspace/projects/{self.pid}/members", headers=stranger,
                                          json={"email": "x@example.com"}).status_code, 404)

    def test_declining_removes_the_invitation(self):
        _, headers, email = new_user()
        member = self.invite(email)
        self.assertEqual(self.client.post(f"/workspace/invitations/{member['id']}/decline", headers=headers).status_code, 204)
        self.assertEqual(self.client.get(f"/workspace/projects/{self.pid}", headers=self.owner).json()["members"], [])

    def test_deleting_the_project_ends_sharing(self):
        viewer, _ = self.join("viewer")
        self.assertEqual(self.client.delete(f"/workspace/projects/{self.pid}", headers=self.owner).status_code, 204)
        self.assertNotIn("Rice N trial", [p["name"] for p in self.client.get("/workspace/projects", headers=viewer).json()])
        # No membership may outlive its project: SQLite can hand the same id to the next project created
        with database.SessionLocal() as db:
            self.assertEqual(db.query(ProjectMember).filter(ProjectMember.project_id == self.pid).count(), 0)


if __name__ == "__main__":
    unittest.main()
