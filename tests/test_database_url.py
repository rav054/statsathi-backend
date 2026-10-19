"""The PostgreSQL driver is named by the application, not left to SQLAlchemy's default.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import unittest

from tests.support import post_csv  # noqa: F401  - must come before other app imports

from app.database import database_url  # noqa: E402

POOLER = "user:pw@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres"


class DatabaseUrlTests(unittest.TestCase):
    def test_a_bare_postgres_url_gets_the_installed_driver(self):
        # SQLAlchemy 2.1 reads a bare postgresql:// as psycopg 3, which this project does not install
        self.assertEqual(database_url("postgresql://" + POOLER), "postgresql+psycopg2://" + POOLER)
        self.assertEqual(database_url("postgres://" + POOLER), "postgresql+psycopg2://" + POOLER)

    def test_an_explicit_driver_is_left_alone(self):
        for url in ("postgresql+psycopg2://" + POOLER, "postgresql+psycopg://" + POOLER):
            self.assertEqual(database_url(url), url)

    def test_sqlite_is_untouched(self):
        self.assertEqual(database_url("sqlite:///./statsathi.db"), "sqlite:///./statsathi.db")


if __name__ == "__main__":
    unittest.main()
