import os
from supabase import create_client, Client
from dotenv import load_dotenv


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Supabase environment variables are not configured.")

    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
