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
    def __init__(self, db_url="sqlite:///legale_bot.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def get_session(self):
        return self.Session()

    def clear(self):
        """Deletes all records from the chunks table."""
        session = self.get_session()
        try:
            session.query(ChunkModel).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
