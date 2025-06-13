import streamlit_authenticator as stauth
import streamlit as st
from datetime import datetime, timedelta
import os

def get_auth_config():
    """
    Get authentication configuration for the recruitment app.
    Reads user credentials from Streamlit secrets or environment variables.
    """
    
    # Use bcrypt to hash passwords manually for compatibility
    import bcrypt
    
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Try to get credentials from Streamlit secrets first
    try:
        # Read from Streamlit secrets
        auth_config = st.secrets["auth"]["users"]
        
        credentials = {
            'usernames': {}
        }
        
        # Build credentials from secrets
        for username, user_data in auth_config.items():
            credentials['usernames'][username] = {
                'name': user_data['name'],
                'password': hash_password(user_data['password']),
                'email': user_data['email']
            }
        
        # If we found valid users from secrets, return them
        if credentials['usernames']:
            return credentials
            
    except Exception as e:
        # If secrets failed, fall back to environment variables
        pass
    
    # Fallback to environment variables
    credentials = {
        'usernames': {
            'admin': {
                'name': os.getenv('AUTH_ADMIN_NAME'),
                'password': hash_password(os.getenv('AUTH_ADMIN_PASSWORD')) if os.getenv('AUTH_ADMIN_PASSWORD') else None,
                'email': os.getenv('AUTH_ADMIN_EMAIL')
            },
            'mithun': {
                'name': os.getenv('AUTH_HR_NAME'),
                'password': hash_password(os.getenv('mithun')) if os.getenv('mithun') else None,
                'email': os.getenv('AUTH_HR_EMAIL')
            },
            'recruiter': {
                'name': os.getenv('AUTH_RECRUITER_NAME'),
                'password': hash_password(os.getenv('AUTH_RECRUITER_PASSWORD')) if os.getenv('AUTH_RECRUITER_PASSWORD') else None,
                'email': os.getenv('AUTH_RECRUITER_EMAIL')
            },
            'ld': {
                'name': os.getenv('AUTH_LD_NAME'),
                'password': hash_password(os.getenv('AUTH_LD_PASSWORD')) if os.getenv('AUTH_LD_PASSWORD') else None,
                'email': os.getenv('AUTH_LD_EMAIL')
            },
        }
    }
    
    # Remove users with missing credentials when using environment variables
    credentials['usernames'] = {
        username: user_data 
        for username, user_data in credentials['usernames'].items() 
        if user_data['name'] and user_data['password'] and user_data['email']
    }
    
    # Check if we have any valid users
    if not credentials['usernames']:
        st.error("❌ **Authentication Error**: No valid user credentials found.")
        st.error("Please set the required AUTH_* environment variables or configure secrets.toml")
        st.stop()
    
    return credentials

def setup_authentication():
    """
    Set up authentication for the Streamlit app.
    Returns the authenticator object and authentication status.
    """
    
    credentials = get_auth_config()
    
    # Get cookie configuration from secrets first, then environment variables
    try:
        # Read from Streamlit secrets
        cookie_name = st.secrets["auth"]["cookie_name"]
        cookie_key = st.secrets["auth"]["cookie_key"]
        cookie_expiry_days = st.secrets["auth"]["cookie_expiry_days"]
    except (KeyError, FileNotFoundError):
        # Fallback to environment variables
        cookie_name = os.getenv('AUTH_COOKIE_NAME', 'saachi_recruitment_app')
        cookie_key = os.getenv('AUTH_COOKIE_KEY')
        cookie_expiry_days = int(os.getenv('AUTH_COOKIE_EXPIRY_DAYS', '7'))
    
    # Validate required cookie key
    if not cookie_key:
        st.error("❌ **Security Error**: Cookie key is required for authentication.")
        st.error("Please set AUTH_COOKIE_KEY in environment variables or configure secrets.toml")
        st.stop()
    
    # Create authenticator object
    authenticator = stauth.Authenticate(
        credentials,
        cookie_name,
        cookie_key,
        cookie_expiry_days=cookie_expiry_days
    )
    
    return authenticator

def create_login_page():
    """
    Create a clean login page for the recruitment app with Saachi AI branding.
    Header at the top center, login form below.
    """
    
    # Header with actual Saachi AI logo - TOP CENTER
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 3rem; text-align: center;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQwIiBoZWlnaHQ9IjE1MSIgdmlld0JveD0iMCAwIDE0MCAxNTEiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNODguNTAzIDQuOTU3ODZDNzcuMDUzMyAtMS42NTI2MiA2Mi45NDY3IC0xLjY1MjYyIDUxLjQ5NyA0Ljk1Nzg1TDE4LjUzMTMgMjMuOTkwNkM3LjA4MTYxIDMwLjYwMTEgMC4wMjgzMjAzIDQyLjgxNzggMC4wMjgzMjAzIDU2LjAzODdWOTQuMTA0MkMwLjAyODMyMDMgMTA3LjMyNSA3LjA4MTYgMTE5LjU0MiAxOC41MzEzIDEyNi4xNTJMNTEuNDk3IDE0NS4xODVDNjIuOTQ2NyAxNTEuNzk2IDc3LjA1MzMgMTUxLjc5NiA4OC41MDI5IDE0NS4xODVMMTIxLjQ2OSAxMjYuMTUyQzEzMi45MTggMTE5LjU0MiAxMzkuOTcyIDEwNy4zMjUgMTM5Ljk3MiA5NC4xMDQyVjU2LjAzODdDMTM5Ljk3MiA0Mi44MTc4IDEzMi45MTggMzAuNjAxMSAxMjEuNDY5IDIzLjk5MDZMODguNTAzIDQuOTU3ODZaTTEwOC41ODcgNTkuNzg1N0M5Ny43MTY1IDU4LjEyMjYgODcuMjkwNCA1Ni40ODgzIDgxLjI0NzIgNTAuNDUyNkM3NS4yMDM0IDQ0LjQxMjUgNzMuNDE4OSAzMy44MzM3IDcxLjU5NzMgMjIuODE1MUM3MS41MjY4IDIyLjM4MjIgNzEuMTQ2NiAyMi4wNzgyIDcwLjcyMDIgMjIuMDg3OEM3MC4yOTM3IDIyLjA3ODIgNjkuOTEzNSAyMi4zODIzIDY5Ljg0MzEgMjIuODE1MUM2OC4wMjE1IDMzLjgzMzcgNjYuMjM3IDQ0LjQxMjUgNjAuMTkzMSA1MC40NTI2QzU0LjE0OSA1Ni40ODkyIDQzLjcyMDggNTguMTIzMSAzMi44NDg5IDU5Ljc4NjRDMzIuMzc1MiA1OS44NTk5IDMyLjA0OTIgNjAuMzA0MiAzMi4xMjI3IDYwLjc3NzlDMzIuMTI5MSA2MC44MjE5IDMyLjEzODcgNjAuODY0NiAzMi4xNTEzIDYwLjkwNTlMMzIuMTUwNyA2MC45MDU3SDMyLjE0NzFMMzIuMTM5NyA2MC45MTMxQzMyLjEyNSA2MC45NDk4IDMyLjExNzcgNjAuOTkwMiAzMi4xMTAzIDYxLjAzMDVDMzIuMDM2OSA2MS41MTE2IDMyLjM2NzQgNjEuOTU5NSAzMi44NDg0IDYyLjAzM0MzNy40NzQ5IDYyLjc0MTYgNDIuMDUgNjMuNDQyOSA0Ni4yNjQ4IDY0LjUwNDFDNTEuNTcwNiA2NS44MzcgNTYuMjA4IDY3LjcxNyA1OS41OTAxIDcwLjc5NzZINTkuNTg2NEM1OS43OTIxIDcwLjk4NDkgNTkuOTkwMyA3MS4xNzIxIDYwLjE4NSA3MS4zNjY3QzY2LjIyODggNzcuNDAzMiA2OC4wMDk2IDg3Ljk3ODIgNjkuODMxMSA5OC45OTc3QzY5LjkwNDUgOTkuNDMxIDcwLjI3NTQgOTkuNzM1NyA3MC42OTc3IDk5LjczNTdINzAuNzA4N1Y5OS43Mjg0SDcwLjcxMjNMNzAuNzE5NyA5OS43MzU3SDcwLjcyMzRINzAuNzM0NEg3MC43NDE3QzcwLjc1NTkgOTkuNzM1NyA3MC43Njg0IDk5LjczNTIgNzAuNzgwNiA5OS43MzQzQzcwLjc5MzggOTkuNzMzMyA3MC44MDY2IDk5LjczMiA3MC44MjA1IDk5LjczMDVMNzAuODQwOSA5OS43Mjg0SDcwLjg0NDVWOTkuNzI0N0M3MC44OTI4IDk5LjcxNjUgNzAuOTM5NSA5OS43MDQ2IDcwLjk4NDUgOTkuNjg5MkM3MS4yOTQ5IDk5LjU5OCA3MS41NDE5IDk5LjMzNzMgNzEuNTk3MyA5OC45OTY4QzcxLjY1MyA5OC42NjAxIDcxLjcwODYgOTguMzIzOCA3MS43NjQzIDk3Ljk4OEg3MS43NjYyTDcxLjc2NDYgOTcuOTg2NEw3MS43NjU1IDk3Ljk4MDZINzEuNzc3MkM3My40NzcyIDg3LjcyNTMgNzUuMjYxNyA3Ny45NjkyIDgwLjYyMjcgNzIuMDMyMkw4MC42MTkgNzIuMDM1OUw4MC42MjI3IDcyLjAxNzZMODAuNjIxOSA3Mi4wMTY4QzgwLjgyNTIgNzEuNzkyMSA4MS4wMzM1IDcxLjU3MjkgODEuMjQ3MiA3MS4zNTkzQzg3LjI5MTMgNjUuMzIyOCA5Ny43MTk2IDYzLjY4ODggMTA4LjU5MSA2Mi4wMjU1QzEwOS4wNjUgNjEuOTUyMSAxMDkuMzkxIDYxLjUwNzcgMTA5LjMxOCA2MS4wMzQxQzEwOS4zMTEgNjAuOTkgMTA5LjMwMiA2MC45NDczIDEwOS4yODkgNjAuOTA2QzEwOS4zMDIgNjAuODY0NyAxMDkuMzExIDYwLjgyMTkgMTA5LjMxOCA2MC43Nzc5QzEwOS4zNTggNjAuNTE5NSAxMDkuMjc5IDYwLjI2OTggMTA5LjEyMiA2MC4wODQ0QzEwOC45OTMgNTkuOTI5NCAxMDguODA4IDU5LjgxOTQgMTA4LjU5MiA1OS43ODY0TDEwOC41ODcgNTkuNzg1N1pNNjUuMTU5OCA3NS4xNjAxQzY1LjE5NDEgNzUuMjIyIDY1LjIyODMgNzUuMjg0IDY1LjI2MjIgNzUuMzQ2MUM2NS4yNTgxIDc1LjMxNTcgNjUuMjI0IDc1LjI1MzggNjUuMTU5OCA3NS4xNjAxWiIgZmlsbD0id2hpdGUiLz4KPC9zdmc+Cg==" alt="Saachi AI Logo" width="40">
                <h1 style="margin: 0; color: #ffffff;">Saachi AI</h1>
            </div>
            <h3 style="color: #ffffff; margin: 0;">Candidate Search Platform</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Add CSS to style the login button
    st.markdown("""
        <style>
        /* Center the login form container */
        .stForm {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        
        /* Target all buttons in the app and override only login-related ones */
        .main button,
        .stApp button,
        button {
            /* Check if button contains Login text and apply styles */
        }
        
        /* More specific targeting for form buttons */
        .main .stForm button,
        .stApp .stForm button,
        div[data-testid="stForm"] button,
        form button {
            display: block !important;
            margin: 0 auto !important;
            width: 200px !important;
            background-color: #dc3545 !important;
            border: 2px solid #dc3545 !important;
            color: white !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            text-align: center !important;
            min-height: 40px !important;
            cursor: pointer !important;
        }
        
        .main .stForm button:hover,
        .stApp .stForm button:hover,
        div[data-testid="stForm"] button:hover,
        form button:hover {
            background-color: #c82333 !important;
            border-color: #bd2130 !important;
        }
        
        /* Specifically exclude the password visibility button */
        .main .stForm button[title*="password"],
        .stApp .stForm button[title*="password"],
        .main .stForm button[aria-label*="password"],
        .stApp .stForm button[aria-label*="password"],
        button[data-testid*="password"],
        button[title*="Show"],
        button[title*="Hide"] {
            background-color: transparent !important;
            border: 1px solid #ccc !important;
            color: inherit !important;
            width: auto !important;
            min-height: auto !important;
            padding: 0.25rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

def show_user_info(name, username):
    """
    Show logged-in user information in the sidebar.
    """
    with st.sidebar:
        # st.success(f"👋 Welcome, **{name}**!")
        # st.caption(f"Logged in as: `{username}`")
        st.markdown("---") 