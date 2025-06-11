import streamlit_authenticator as stauth
import streamlit as st
from datetime import datetime, timedelta

def get_auth_config():
    """
    Get authentication configuration for the recruitment app.
    This creates a simple user database with pre-hashed passwords.
    """
    
    # Use bcrypt to hash passwords manually for compatibility
    import bcrypt
    
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Create credentials dictionary with hashed passwords
    credentials = {
        'usernames': {
            'admin': {
                'name': 'Administrator',
                'password': hash_password('admin1@apples'),
                'email': 'admin@company.com'
            },
            'hr_manager': {
                'name': 'HR Manager', 
                'password': hash_password('hr2024@oranges'),
                'email': 'hr@company.com'
            },
            'recruiter': {
                'name': 'Recruiter',
                'password': hash_password('recruit2024@bananas'),
                'email': 'recruiter@company.com'
            },
            'LD' or 'ld': {
                'name': 'Lakshman',
                'password': hash_password('lakshman@tomatoes'),
                'email': 'lakshman@company.com'
            },
        }
    }
    
    return credentials

def setup_authentication():
    """
    Set up authentication for the Streamlit app.
    Returns the authenticator object and authentication status.
    """
    
    credentials = get_auth_config()
    
    # Create authenticator object
    authenticator = stauth.Authenticate(
        credentials,
        'saachi_recruitment_app',  # cookie name
        'recruitment_secret_key_2024',  # cookie key (change this to something secure)
        cookie_expiry_days=7  # Cookie expires in 7 days
    )
    
    return authenticator

def create_login_page():
    """
    Create a beautiful login page for the recruitment app.
    """
    
    # Custom CSS for the login page
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            text-align: center;
            margin-top: 3rem;
        }
        .login-title {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .login-subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .demo-credentials {
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Login page header
    st.markdown("""
        <div class="login-container">
            <div class="login-title">🔍 Saachi AI</div>
            <div class="login-subtitle">Candidate Search Platform</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions
    st.info("🔐 Please login to access the AI-powered candidate search system")

def show_user_info(name, username):
    """
    Show logged-in user information in the sidebar.
    """
    with st.sidebar:
        st.success(f"👋 Welcome, **{name}**!")
        st.caption(f"Logged in as: `{username}`")
        st.markdown("---") 