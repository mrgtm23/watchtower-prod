import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func
from backend.app.db.base_class import Base

class User(Base):
    __tablename__ = "users"
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelArtifact(Base):
    __tablename__ = "models"
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    owner_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    framework = Column(String, nullable=False)
    artifact_path = Column(String, nullable=False)
    sha256 = Column(String, nullable=False)
    status = Column(String, default="registered")
    meta_info = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ReferenceDataset(Base):
    __tablename__ = "reference_datasets"
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    owner_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    # S3 Path: s3://<bucket>/<owner_id>/datasets/<id>_<name>.csv
    artifact_path = Column(String, nullable=False) 
    sha256 = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())