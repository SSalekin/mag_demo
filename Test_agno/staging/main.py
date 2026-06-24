from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

declaration = declarative_base()

class User(declaration):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

def create_db():
    engine = create_engine('sqlite:///test.db')
    declaration.metadata.create_all(bind=engine)

if __name__ == '__main__':
    print('Database created.')
    create_db()