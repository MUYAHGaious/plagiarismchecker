"""
Database models for the plagiarism checker application.
"""

from datetime import datetime
import uuid

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declared_attr

# Database URL - update for production
DATABASE_URL = "sqlite:///./plagiarism_checker.db"

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Document(Base):
    """Model for uploaded documents."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.now)
    size = Column(Integer)  # File size in bytes
    status = Column(String, default="uploaded")  # uploaded, processing, processed, error
    
    # Relationships
    results = relationship("PlagiarismResult", back_populates="document")
    
    def __repr__(self):
        return f"<Document {self.filename}>"

class PlagiarismResult(Base):
    """Model for plagiarism check results."""
    __tablename__ = "plagiarism_results"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    overall_similarity = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    document = relationship("Document", back_populates="results")
    matches = relationship("PlagiarismMatch", back_populates="result")
    
    def __repr__(self):
        return f"<PlagiarismResult {self.id} - {self.overall_similarity}>"

class PlagiarismMatch(Base):
    """Model for individual document matches within a plagiarism result."""
    __tablename__ = "plagiarism_matches"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    result_id = Column(String, ForeignKey("plagiarism_results.id"), nullable=False)
    matched_document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    similarity_score = Column(Float, default=0.0)
    
    # Relationships
    result = relationship("PlagiarismResult", back_populates="matches")
    matched_document = relationship("Document")
    segments = relationship("MatchedSegment", back_populates="match")
    
    def __repr__(self):
        return f"<PlagiarismMatch {self.id} - {self.similarity_score}>"

class MatchedSegment(Base):
    """Model for matched text segments within a plagiarism match."""
    __tablename__ = "matched_segments"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    match_id = Column(String, ForeignKey("plagiarism_matches.id"), nullable=False)
    source_segment = Column(Text, nullable=False)
    matched_segment = Column(Text, nullable=False)
    similarity = Column(Float, default=0.0)
    source_index = Column(Integer)
    matched_index = Column(Integer)
    
    # Relationships
    match = relationship("PlagiarismMatch", back_populates="segments")
    
    def __repr__(self):
        return f"<MatchedSegment {self.id} - {self.similarity}>"

# Optional: Document vector model for vector database integration
class DocumentVector(Base):
    """Model for document vectors used in similarity search."""
    __tablename__ = "document_vectors"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    vector_id = Column(String, nullable=False)  # ID in the vector database
    text_chunk = Column(Text, nullable=False)
    
    # Relationship
    document = relationship("Document")
    
    def __repr__(self):
        return f"<DocumentVector {self.id} - {self.document_id}>"