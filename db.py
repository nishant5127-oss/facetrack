# Import the operating system module to interact with the file system (e.g., deleting files)
import os

# Import datetime utilities to handle timestamps with UTC timezone awareness
from datetime import datetime, timezone

# Import Optional type hint for fields that can be None (like primary keys before insertion)
from typing import Optional

# Import SQLModel components: base model, table field options, session manager, engine builder, query select, and column helpers
from sqlmodel import SQLModel, Field, Session, create_engine, select, col

# Create the SQLite database engine pointing to local file 'missing.db'
# 'check_same_thread=False' allows multiple Streamlit threads/workers to access SQLite concurrently
engine = create_engine("sqlite:///missing.db", connect_args={"check_same_thread": False})

# Helper lambda function that returns the current UTC timestamp for consistent timestamping
_now = lambda: datetime.now(timezone.utc)


# ============================================================================
# TABLE MODEL: Case (Stores Registered Missing Person Profiles)
# ============================================================================
class Case(SQLModel, table=True):
    # Primary Key: Unique integer ID auto-incremented by SQLite (default=None before DB insertion)
    id: Optional[int] = Field(default=None, primary_key=True)

    # Full name of the missing person (required field)
    name: str

    # Age in years (optional integer, can be None if unknown)
    age: Optional[int] = None

    # Biological gender (e.g., 'Male', 'Female', 'Other')
    gender: str = ""

    # Last seen city or location where the person went missing
    city: str = ""

    # Contact email of the investigating officer or family member for alert emails
    contact_email: str = ""

    # Text description with physical traits, clothing, or case background
    description: str = ""

    # Local disk file path to the reference photo in the 'uploads/' folder
    photo_path: str = ""

    # 1404-dimensional facial landmark embedding serialized as a JSON string
    embedding: str = ""            # JSON list of 1404 floats

    # Status of the case: 'missing' (active investigation) or 'found' (case closed)
    status: str = "missing"        # missing | found

    # Boolean flag: True if displayed publicly on citizen board; False for private police record
    published: bool = False        # shown on the public board

    # Timestamp when the case was first registered (defaults to current UTC time)
    created_at: datetime = Field(default_factory=_now)


# ============================================================================
# TABLE MODEL: Sighting (Stores Potential Sightings Submitted by Public/CCTV)
# ============================================================================
class Sighting(SQLModel, table=True):
    # Primary Key: Unique integer ID for each sighting record
    id: Optional[int] = Field(default=None, primary_key=True)

    # Foreign Key: References the 'id' column of the 'case' table (links sighting to missing person)
    case_id: Optional[int] = Field(default=None, foreign_key="case.id")

    # City where the sighting was reported or captured
    city: str = ""

    # Geographic Latitude coordinate for map visualization in Folium
    lat: Optional[float] = None

    # Geographic Longitude coordinate for map visualization in Folium
    lon: Optional[float] = None

    # Email address of the citizen reporting the sighting (optional)
    reporter_email: str = ""

    # Local disk file path to the sighting photo in the 'uploads/' folder
    photo_path: str = ""

    # Match confidence percentage calculated by matcher.py (0.0% to 100.0%)
    confidence: float = 0.0

    # Human-in-the-loop review state: 'pending' (new), 'confirmed' (verified), or 'dismissed' (rejected)
    review_status: str = "pending"  # pending | confirmed | dismissed

    # Timestamp when the sighting report was submitted
    created_at: datetime = Field(default_factory=_now)


# ============================================================================
# HELPER & DATABASE OPERATIONS
# ============================================================================

def _rm(path):
    """Safely removes a file from the disk if the path exists, ignoring OS errors."""
    try:
        # Check if path string is non-empty before calling os.remove
        if path: os.remove(path)
    except OSError:
        # Silently pass if file does not exist or cannot be accessed
        pass


def init():
    """Initializes the database: creates tables, runs migrations, and purges orphan files."""
    # Create all tables defined by SQLModel metadata if they do not exist
    SQLModel.metadata.create_all(engine)

    # Lightweight migration scripts to add newly added columns to existing database schemas
    for stmt in ("ALTER TABLE sighting ADD COLUMN review_status VARCHAR DEFAULT 'pending'",
                 'ALTER TABLE "case" ADD COLUMN published BOOLEAN DEFAULT 0'):
        try:
            # Begin a transactional database connection
            with engine.begin() as conn:
                # Execute direct SQL statements for table alteration
                conn.exec_driver_sql(stmt)
        except Exception:
            # Ignore errors if columns already exist in missing.db
            pass

    # Storage Minimization (DPDP Compliance): purge any orphan sighting uploads without a matched case
    with session() as s:  # unmatched uploads are no longer kept
        # Query all sightings where case_id is None
        for x in s.exec(select(Sighting).where(col(Sighting.case_id).is_(None))):
            # Delete physical image file from disk
            _rm(x.photo_path)
            # Delete database row from SQLite
            s.delete(x)
        # Commit cleanup changes to database
        s.commit()


def session():
    """Creates and returns a new SQLModel database session connected to engine."""
    return Session(engine)


def active_cases():
    """Fetches all registered cases currently marked with status 'missing'."""
    with session() as s:
        # Execute SELECT query filtering where status == 'missing'
        return list(s.exec(select(Case).where(Case.status == "missing")))


def published_cases():
    """Fetches missing cases marked as published for display on public website board."""
    with session() as s:
        # Filter for missing and published=True, ordered newest to oldest
        return list(s.exec(select(Case).where(Case.status == "missing", Case.published == True)  # noqa: E712
                           .order_by(Case.created_at.desc())))


def all_cases():
    """Fetches all cases (both missing and found) ordered by registration date descending."""
    with session() as s:
        # Execute query selecting all cases ordered newest first
        return list(s.exec(select(Case).order_by(Case.created_at.desc())))


def all_sightings():
    """Fetches all reported sightings ordered by submission date descending."""
    with session() as s:
        # Execute query selecting all sightings ordered newest first
        return list(s.exec(select(Sighting).order_by(Sighting.created_at.desc())))


def add(obj):
    """Inserts a new Case or Sighting object, commits transaction, and refreshes object with new ID."""
    with session() as s:
        # Add entity to active session
        s.add(obj)
        # Commit transaction to disk
        s.commit()
        # Refresh object attributes (e.g. populates auto-generated primary key 'id')
        s.refresh(obj)
    # Return the persisted object with updated ID
    return obj


def mark_found(case_id: int):
    """Updates case status to 'found' (closing the active investigation)."""
    with session() as s:
        # Fetch case record by its primary key
        c = s.get(Case, case_id)
        # If record found in database
        if c:
            # Change status to 'found'
            c.status = "found"
            # Stage updated case
            s.add(c)
            # Commit update
            s.commit()


def set_published(case_id: int, value: bool):
    """Toggles whether a case is published on the public missing person board."""
    with session() as s:
        # Fetch case record by ID
        c = s.get(Case, case_id)
        # If record found
        if c:
            # Set published flag to True or False
            c.published = value
            # Stage record
            s.add(c)
            # Commit update
            s.commit()


def set_review(sighting_id: int, status: str):
    """Updates sighting review status ('confirmed' or 'dismissed') by investigating officer."""
    with session() as s:
        # Fetch sighting record by primary key
        x = s.get(Sighting, sighting_id)
        # If sighting exists
        if x:
            # Update review status column
            x.review_status = status
            # Stage update
            s.add(x)
            # Commit change
            s.commit()


def delete_case(case_id: int):
    """Deletes a case, all associated sightings, and permanently removes photo files from disk."""
    # List to collect file paths for deletion after database transaction
    paths = []
    with session() as s:
        # Find all sightings linked to this case_id
        for x in s.exec(select(Sighting).where(Sighting.case_id == case_id)):
            # Record photo path for disk deletion
            paths.append(x.photo_path)
            # Delete sighting row from DB
            s.delete(x)
        # Fetch the main case record
        c = s.get(Case, case_id)
        # If case exists
        if c:
            # Record reference photo path
            paths.append(c.photo_path)
            # Delete case row from DB
            s.delete(c)
        # Commit database transaction
        s.commit()
    # Iterate through all collected file paths
    for p in paths:
        # Delete file from the uploads directory
        _rm(p)
