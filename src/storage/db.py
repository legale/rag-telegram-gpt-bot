from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class ChunkModel(Base):
    __tablename__ = 'chunks'
    
    id = Column(String, primary_key=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Store metadata as JSON string or separate columns if needed
    metadata_json = Column(Text, nullable=True)

class Database:
    def __init__(self, db_url: str):
        if not db_url:
            raise ValueError("db_url must be provided")
        self.db_url = db_url
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def get_session(self):
        return self.Session()

    def count_chunks(self) -> int:
        """Returns number of stored chunks."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).count()
        finally:
            session.close()

    def clear(self) -> int:
        """Deletes all records from the chunks table and returns removed count."""
        session = self.get_session()
        try:
            deleted = session.query(ChunkModel).delete()
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
