# Student Modeling

Graph-based student modeling system using Neo4j. Tracks students, courses, skills, and knowledge states as a connected graph.

## Setup

```bash
# Start Neo4j
docker compose up -d

# Install (editable)
cp .env.example .env
pip install -e ".[dev]"
```

## Graph Schema

```
(:Student)-[:ENROLLED_IN]->(:Course)
(:Course)-[:TEACHES]->(:Skill)
(:Course)-[:REQUIRES]->(:Skill)
(:Skill)-[:PREREQUISITE_OF]->(:Skill)
(:Student)-[:HAS_KNOWLEDGE {mastery_level, confidence}]->(:Skill)
```

## Usage

```python
from student_modeling.config import get_settings
from student_modeling.database import Database

settings = get_settings()
Database.connect(settings)

# Use repositories for data access
from student_modeling.repositories.student_repository import StudentRepository

repo = StudentRepository(Database.get_driver(), settings.neo4j_database)
```

## Tests

```bash
# Requires running Neo4j
pytest
```
