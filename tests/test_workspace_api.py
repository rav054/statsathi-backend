"""Workspace API: projects, stored datasets and saved analysis results.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import io
import unittest

from tests.support import client_and_headers  # must come before other app imports

import pandas as pd  # noqa: E402

from app import workspace_routes  # noqa: E402

CSV = pd.DataFrame({"Treatment": ["T1", "T2", "T3", "T4"], "Yield": [41.2, 44.8, 39.5, 47.1]}).to_csv(index=False).encode()
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class WorkspaceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, cls.headers = client_and_headers("workspace-owner@example.com")
        cls.other_headers = client_and_headers("workspace-other@example.com")[1]

    def api(self, method, path, headers=None, **kwargs):
        return self.client.request(method, f"/workspace{path}", headers=headers or self.headers, **kwargs)

    def create_project(self, name="Rice split-plot"):
        response = self.api("POST", "/projects", json={"name": name, "tasks": [{"id": 1, "text": "Upload data"}]})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def upload(self, project_id, content=CSV, filename="yield.csv", media="text/csv", headers=None):
        return self.api("POST", f"/projects/{project_id}/datasets", headers=headers, files={"file": (filename, content, media)})

    def save_result(self, project_id, **extra):
        body = {"module": "anova", "title": "Yield ANOVA", "parameters": {"test_type": "oneway"},
                "result": {"anova_table": {"p_value": 0.012}}, **extra}
        return self.api("POST", f"/projects/{project_id}/results", json=body)

    def test_new_user_gets_the_sample_project(self):
        _, headers = client_and_headers("workspace-newcomer@example.com")
        projects = self.api("GET", "/projects", headers=headers).json()
        self.assertEqual([p["name"] for p in projects], ["Dummy project"])
        self.assertEqual(len(projects[0]["tasks"]), 4)

    def test_project_crud_and_status_validation(self):
        project = self.create_project()
        updated = self.api("PATCH", f"/projects/{project['id']}", json={"name": "Renamed", "status": "In Progress"})
        self.assertEqual((updated.json()["name"], updated.json()["status"]), ("Renamed", "In Progress"))
        self.assertEqual(self.api("PATCH", f"/projects/{project['id']}", json={"status": "Finished"}).status_code, 422)
        self.assertEqual(self.api("DELETE", f"/projects/{project['id']}").status_code, 204)
        self.assertEqual(self.api("GET", f"/projects/{project['id']}").status_code, 404)

    def test_other_users_get_404_for_everything(self):
        project = self.create_project()
        dataset = self.upload(project["id"]).json()
        result = self.save_result(project["id"]).json()
        requests = [
            ("GET", f"/projects/{project['id']}", {}),
            ("PATCH", f"/projects/{project['id']}", {"json": {"name": "Hijacked"}}),
            ("DELETE", f"/projects/{project['id']}", {}),
            ("GET", f"/datasets/{dataset['id']}/download", {}),
            ("DELETE", f"/datasets/{dataset['id']}", {}),
            ("GET", f"/results/{result['id']}", {}),
            ("DELETE", f"/results/{result['id']}", {}),
        ]
        for method, path, kwargs in requests:
            self.assertEqual(self.api(method, path, headers=self.other_headers, **kwargs).status_code, 404, (method, path))
        self.assertEqual(self.upload(project["id"], headers=self.other_headers).status_code, 404)
        self.assertEqual(self.api("GET", f"/projects/{project['id']}").json()["name"], "Rice split-plot")

    def test_csv_dataset_round_trip_and_metadata(self):
        project = self.create_project()
        meta = self.upload(project["id"]).json()
        self.assertEqual((meta["n_rows"], meta["n_columns"], meta["columns"]), (4, 2, ["Treatment", "Yield"]))
        download = self.api("GET", f"/datasets/{meta['id']}/download")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.content, CSV)
        self.assertIn("yield.csv", download.headers["content-disposition"])
        detail = self.api("GET", f"/projects/{project['id']}").json()
        self.assertEqual(detail["dataset_count"], 1)
        self.assertEqual((detail["samples"], detail["variables"]), (4, 2))

    def test_excel_dataset(self):
        project = self.create_project()
        buffer = io.BytesIO()
        pd.read_csv(io.BytesIO(CSV)).to_excel(buffer, index=False)
        response = self.upload(project["id"], buffer.getvalue(), "yield.xlsx", XLSX_TYPE)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["n_rows"], 4)
        self.assertEqual(self.api("GET", f"/datasets/{response.json()['id']}/download").content, buffer.getvalue())

    def test_unreadable_and_oversized_files_are_rejected(self):
        project = self.create_project()
        self.assertEqual(self.upload(project["id"], b"not a table", "notes.txt", "text/plain").status_code, 400)
        original = workspace_routes.MAX_DATASET_BYTES
        workspace_routes.MAX_DATASET_BYTES = 10
        try:
            self.assertEqual(self.upload(project["id"]).status_code, 413)
        finally:
            workspace_routes.MAX_DATASET_BYTES = original

    def test_saved_results_unlink_from_deleted_datasets_and_cascade_with_projects(self):
        project = self.create_project()
        dataset = self.upload(project["id"]).json()
        saved = self.save_result(project["id"], dataset_id=dataset["id"])
        self.assertEqual(saved.status_code, 201, saved.text)
        result_id = saved.json()["id"]
        full = self.api("GET", f"/results/{result_id}").json()
        self.assertEqual(full["result"], {"anova_table": {"p_value": 0.012}})

        self.assertEqual(self.api("DELETE", f"/datasets/{dataset['id']}").status_code, 204)
        self.assertIsNone(self.api("GET", f"/results/{result_id}").json()["dataset_id"])

        self.assertEqual(self.api("DELETE", f"/projects/{project['id']}").status_code, 204)
        self.assertEqual(self.api("GET", f"/results/{result_id}").status_code, 404)

    def test_result_dataset_must_belong_to_the_same_project(self):
        first, second = self.create_project("First"), self.create_project("Second")
        dataset = self.upload(first["id"]).json()
        self.assertEqual(self.save_result(second["id"], dataset_id=dataset["id"]).status_code, 400)

    def test_import_browser_projects(self):
        response = self.api("POST", "/import", json={"projects": [
            {"name": "Dummy project", "tasks": []},
            {"name": "Wheat RBD", "created": "2026-05-01", "status": "Completed", "samples": 60, "variables": 5,
             "testUsed": "RBD ANOVA", "tasks": [{"id": 7, "text": "Run DMRT", "completed": True}]},
        ]})
        self.assertEqual(response.json(), {"imported": 1})
        wheat = next(p for p in self.api("GET", "/projects").json() if p["name"] == "Wheat RBD")
        self.assertTrue(wheat["created_at"].startswith("2026-05-01"))
        self.assertEqual((wheat["status"], wheat["test_used"], wheat["tasks"][0]["text"]), ("Completed", "RBD ANOVA", "Run DMRT"))

    def test_storage_summary_counts_uploads(self):
        project = self.create_project()
        before = self.api("GET", "/storage").json()["used_bytes"]
        self.upload(project["id"])
        self.assertEqual(self.api("GET", "/storage").json()["used_bytes"] - before, len(CSV))

    RECIPE = {"name": "Kharif cleaning", "source_dataset": "kharif2026.csv",
              "steps": [{"type": "drop_duplicates"}, {"type": "fill_missing", "columns": ["Yield"], "method": "median"}]}

    def test_recipes_are_stored_listed_replaced_and_deleted(self):
        project = self.create_project("Recipe project")
        saved = self.api("POST", f"/projects/{project['id']}/recipes", json=self.RECIPE)
        self.assertEqual(saved.status_code, 201, saved.text)
        recipe = saved.json()
        self.assertEqual((recipe["step_count"], recipe["source_dataset"]), (2, "kharif2026.csv"))
        self.assertGreater(recipe["size_bytes"], 0)

        detail = self.api("GET", f"/projects/{project['id']}").json()
        self.assertEqual([r["name"] for r in detail["recipes"]], ["Kharif cleaning"])
        listed = [r for r in self.api("GET", "/recipes").json() if r["id"] == recipe["id"]]
        self.assertEqual(listed[0]["project_name"], "Recipe project")

        changed = {**self.RECIPE, "steps": [{"type": "drop_duplicates"}]}
        replaced = self.api("PUT", f"/recipes/{recipe['id']}", json=changed).json()
        self.assertEqual((replaced["id"], replaced["step_count"]), (recipe["id"], 1))
        self.assertEqual(self.api("GET", f"/recipes/{recipe['id']}").json()["steps"], [{"type": "drop_duplicates"}])

        self.assertEqual(self.api("DELETE", f"/recipes/{recipe['id']}").status_code, 204)
        self.assertEqual(self.api("GET", f"/recipes/{recipe['id']}").status_code, 404)

    def test_recipes_are_validated_private_and_counted(self):
        project = self.create_project("Recipe checks")
        path = f"/projects/{project['id']}/recipes"
        self.assertEqual(self.api("POST", path, json={**self.RECIPE, "steps": []}).status_code, 422)
        self.assertEqual(self.api("POST", path, json={**self.RECIPE, "steps": [{"type": "teleport"}]}).status_code, 422)

        before = self.api("GET", "/storage").json()["used_bytes"]
        recipe = self.api("POST", path, json=self.RECIPE).json()
        self.assertEqual(self.api("GET", "/storage").json()["used_bytes"] - before, recipe["size_bytes"])

        self.assertEqual(self.api("GET", f"/recipes/{recipe['id']}", headers=self.other_headers).status_code, 404)
        self.assertEqual(self.api("POST", path, headers=self.other_headers, json=self.RECIPE).status_code, 404)
        self.assertNotIn(recipe["id"], [r["id"] for r in self.api("GET", "/recipes", headers=self.other_headers).json()])

        # recipes go with their project
        self.api("DELETE", f"/projects/{project['id']}")
        self.assertEqual(self.api("GET", f"/recipes/{recipe['id']}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
