from sqlalchemy import Column, String, Float, DateTime, Boolean, ForeignKey, Text, Integer
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String)
    file_path = Column(String)
    upload_date = Column(DateTime, default=datetime.now)
    size = Column(Integer)
    status = Column(String, default="uploaded")  # uploaded, processing, processed, error
    
    # Relationships
    results = relationship("PlagiarismResult", back_populates="document", cascade="all, delete-orphan")

class PlagiarismResult(Base):
    __tablename__ = "plagiarism_results"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    overall_similarity = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    document = relationship("Document", back_populates="results")
    matches = relationship("PlagiarismMatch", back_populates="result", cascade="all, delete-orphan")

class PlagiarismMatch(Base):
    __tablename__ = "plagiarism_matches"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    result_id = Column(String, ForeignKey("plagiarism_results.id"))
    matched_document_id = Column(String, ForeignKey("documents.id"))
    similarity_score = Column(Float)
    
    # Relationships
    result = relationship("PlagiarismResult", back_populates="matches")
    matched_document = relationship("Document", foreign_keys=[matched_document_id])
    segments = relationship("MatchedSegment", back_populates="match", cascade="all, delete-orphan")

class MatchedSegment(Base):
    __tablename__ = "matched_segments"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    match_id = Column(String, ForeignKey("plagiarism_matches.id"))
    source_segment = Column(Text)
    matched_segment = Column(Text)
    similarity = Column(Float)
    source_index = Column(Integer, nullable=True)
    matched_index = Column(Integer, nullable=True)
    
    # Relationships
    match = relationship("PlagiarismMatch", back_populates="segments")