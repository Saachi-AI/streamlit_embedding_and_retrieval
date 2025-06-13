# Security Setup Guide

## Authentication Configuration

The Saachi AI recruitment platform now uses secure credential management instead of hardcoded passwords. This guide explains how to set up authentication for different deployment scenarios.

## 🔐 Security Improvements

✅ **No hardcoded credentials in code** - All sensitive data externalized from Python files  
✅ **Streamlit secrets support** - Uses `secrets.toml` for Streamlit Cloud deployment  
✅ **Environment variable fallback** - Works with traditional env vars for other platforms  
✅ **Clean codebase** - Zero credentials in the actual application code  

## 📋 Setup Instructions

### For Local Development

1. **Copy the environment template:**
   ```bash
   cp env_template.txt .env
   ```

2. **Edit `.env` file with your credentials:**
   - Change all `your_*_here` placeholders to actual values
   - Use strong, unique passwords for each user
   - Update email addresses to match your organization

3. **The app will automatically load from `.env` file**

### For Streamlit Community Cloud

1. **Update the `secrets.toml` file with your credentials:**
   ```toml
   [auth]
   cookie_key = "your_secure_cookie_key_here"
   
   [auth.users.admin]
   password = "your_secure_admin_password"
   # ... update all user passwords
   ```

2. **Upload to Streamlit Cloud's secret manager:**
   - The `secrets.toml` file will be automatically used by Streamlit Cloud
   - Never commit this file to your repository (it's in .gitignore)

### For Production Deployment

1. **Set environment variables on your server:**
   ```bash
   export AUTH_ADMIN_PASSWORD="your_secure_password"
   export AUTH_COOKIE_KEY="your_secure_cookie_key"
   # ... set all required variables
   ```

2. **Or use your platform's secret management:**
   - AWS Secrets Manager
   - Azure Key Vault  
   - Google Secret Manager
   - Docker secrets
   - Kubernetes secrets

## 🔑 Required Authentication Variables

### Cookie Configuration
- `AUTH_COOKIE_NAME` - Name for authentication cookie
- `AUTH_COOKIE_KEY` - Secret key for cookie encryption (change this!)
- `AUTH_COOKIE_EXPIRY_DAYS` - Cookie expiration in days

### User Credentials (for each user: admin, hr_manager, recruiter, ld)
- `AUTH_{USER}_NAME` - Display name
- `AUTH_{USER}_PASSWORD` - Login password  
- `AUTH_{USER}_EMAIL` - Email address

Example:
```bash
AUTH_ADMIN_NAME="Administrator"
AUTH_ADMIN_PASSWORD="secure_admin_password_123"
AUTH_ADMIN_EMAIL="admin@yourcompany.com"
```

## 🛡️ Security Best Practices

1. **Change default passwords** - Never use the demo passwords in production
2. **Use strong passwords** - Minimum 12 characters with mixed case, numbers, symbols
3. **Rotate credentials regularly** - Update passwords periodically
4. **Secure cookie key** - Use a long, random string for `AUTH_COOKIE_KEY`
5. **HTTPS only** - Always deploy with SSL/TLS in production
6. **Monitor access** - Review authentication logs regularly

## 🔄 How It Works

The authentication system now:

1. **First tries** to load from Streamlit `secrets.toml` (for Streamlit Cloud)
2. **Falls back** to environment variables (for other deployments)
3. **Validates all credentials** - app won't start if configuration is incomplete
4. **Zero hardcoded values** - no credentials anywhere in the Python code

This provides the best of both worlds: Streamlit Cloud compatibility with environment variable flexibility.

## 🚨 Important Notes

- **Never commit** `.env` or `secrets.toml` files to version control
- **Always change** the default `AUTH_COOKIE_KEY` in production
- **Use HTTPS** in production to protect authentication cookies
- **Review user access** regularly and remove unused accounts 