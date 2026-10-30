"""Тести конфігурації розгортання (Dockerfile, docker-compose.yml, README.md)."""

import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestDeploymentConfig(unittest.TestCase):
    """Перевірка наявності та ключових вимог до файлів Docker-розгортання."""

    def test_dockerfile_requirements(self) -> None:
        dockerfile_path = ROOT_DIR / "Dockerfile"
        self.assertTrue(dockerfile_path.is_file(), "Dockerfile має існувати")
        content = dockerfile_path.read_text(encoding="utf-8")
        self.assertIn("FROM python:3.11-slim", content)
        self.assertIn("build-essential", content)
        self.assertIn("libmagic1", content)
        self.assertIn("bot/main.py", content)

    def test_docker_compose_requirements(self) -> None:
        compose_path = ROOT_DIR / "docker-compose.yml"
        self.assertTrue(compose_path.is_file(), "docker-compose.yml має існувати")
        content = compose_path.read_text(encoding="utf-8")
        self.assertIn("bot:", content)
        self.assertIn("./data:/app/data", content)
        self.assertIn(".env", content)

    def test_readme_requirements(self) -> None:
        readme_path = ROOT_DIR / "README.md"
        self.assertTrue(readme_path.is_file(), "README.md має існувати")
        content = readme_path.read_text(encoding="utf-8")
        self.assertIn(".env", content)
        self.assertIn("docker compose up -d --build", content)

    def test_requirements_txt_includes_sqlalchemy_asyncio(self) -> None:
        """SQLAlchemy asyncio потребує пакету greenlet (SQLAlchemy[asyncio])."""
        req_path = ROOT_DIR / "requirements.txt"
        content = req_path.read_text(encoding="utf-8").lower()
        self.assertIn("sqlalchemy[asyncio]", content)


if __name__ == "__main__":
    unittest.main()
