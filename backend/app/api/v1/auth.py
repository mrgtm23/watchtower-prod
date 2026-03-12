from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app import db as db_models
from backend.app.schemas.user import UserCreate, UserOut
from backend.app.core.security import get_password_hash, create_access_token, verify_password
from backend.app.db import models
from backend.app.api.v1.uploads import get_current_user_id

router = APIRouter()

@router.post("/register", response_model=UserOut)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == user_in.email).first():
        raise HTTPException(400, "Email already registered")
    if db.query(models.User).filter(models.User.username == user_in.username).first():
        raise HTTPException(400, "Username taken")
    user = models.User(username=user_in.username, email=user_in.email, hashed_password=get_password_hash(user_in.password))
    db.add(user); db.commit(); db.refresh(user)
    return user

@router.post("/token")
def login_for_access_token(form_data: UserCreate, db: Session = Depends(get_db)):
    # For simplicity, accept JSON with email + password
    user = db.query(models.User).filter(models.User.email == form_data.email).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=UserOut, tags=["users"])
def read_users_me(
    current_user_id: str = Depends(get_current_user_id), # Requires a valid JWT token
    db: Session = Depends(get_db)
):
    """
    Retrieves the authenticated user's profile information using the JWT token.
    The response includes the user's UUID (id).
    """
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    
    # This check is technically redundant since get_current_user_id checks it, 
    # but serves as a good final sanity check.
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    
    return user