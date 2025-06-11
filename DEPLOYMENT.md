# 🚀 Deployment Guide - Streamlit Community Cloud

This guide will help you deploy your Saachi AI Recruitment app to Streamlit Community Cloud with authentication.

## 📋 Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Streamlit Community Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **API Keys** - Have all your API keys ready

## 🔧 Step 1: Prepare Your Repository

### Push to GitHub
```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit with authentication"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push to GitHub
git push -u origin main
```

## 🎯 Step 2: Deploy to Streamlit Community Cloud

### 1. Go to Streamlit Community Cloud
- Visit [share.streamlit.io](https://share.streamlit.io)
- Sign in with your GitHub account

### 2. Create New App
- Click "New app"
- Select your GitHub repository
- Choose the branch (usually `main`)
- Set main file path: `app.py`
- Choose a custom URL (optional)

### 3. Configure Secrets
After creating the app, go to **App Settings** → **Secrets** and add:

```toml
# Pinecone Configuration
PINECONE_API_KEY = "your-actual-pinecone-api-key"
PINECONE_INDEX_NAME = "your-actual-index-name"
PINECONE_ENVIRONMENT = "us-east-1"

# OpenAI Configuration
OPENAI_API_KEY = "your-actual-openai-api-key"

# Cohere Configuration  
COHERE_API_KEY = "your-actual-cohere-api-key"

# Groq Configuration
GROQ_API_KEY = "your-actual-groq-api-key"

# Upstage Configuration
UPSTAGE_API_KEY = "your-actual-upstage-api-key"

# X AI Configuration
XAI_API_KEY = "your-actual-xai-api-key"

# Google Configuration
GOOGLE_API_KEY = "your-actual-google-api-key"

# AWS Configuration
AWS_ACCESS_KEY_ID = "your-actual-aws-access-key"
AWS_SECRET_ACCESS_KEY = "your-actual-aws-secret-key"
AWS_REGION = "ap-northeast-1"

# DynamoDB Tables
TAMAGO_DYNAMODB_TABLE = "tamago_profiles"
LINKEDIN_DYNAMODB_TABLE = "linkedin_profiles"
```

## 🔐 Step 3: Test Authentication

### Demo Login Credentials
Once deployed, your employees can use these credentials:

**Administrator**
- Username: `admin`
- Password: `admin123`

**HR Manager**
- Username: `hr_manager`
- Password: `hr2024`

**Recruiter**
- Username: `recruiter`
- Password: `recruit2024`

## 🛠️ Step 4: Customize Authentication

### Add More Users
Edit `auth_config.py` to add more users:

```python
users = {
    'admin': {
        'name': 'Administrator',
        'password': 'admin123',
        'email': 'admin@company.com'
    },
    'your_employee': {
        'name': 'Employee Name',
        'password': 'secure_password',
        'email': 'employee@company.com'
    },
    # Add more users here...
}
```

### Security Best Practices
1. **Change default passwords** in production
2. **Use strong passwords** (12+ characters, mixed case, numbers, symbols)
3. **Regular password updates** for security
4. **Monitor access logs** through Streamlit Cloud dashboard

## 📱 Step 5: Share with Employees

### Access Instructions for Employees
1. **Send them the app URL** (e.g., `https://your-app-name.streamlit.app`)
2. **Provide login credentials** for their role
3. **Basic usage training** on the AI recruitment features

### Sample Email Template
```
Subject: New AI Recruitment Tool - Saachi AI

Hi Team,

We've deployed our new AI-powered candidate search platform!

🔗 Access URL: https://your-app-name.streamlit.app

🔐 Your Login Credentials:
Username: [their_username]
Password: [their_password]

The platform allows you to:
- Upload job descriptions for automatic candidate matching
- Perform custom candidate searches
- Get AI-powered evaluations and rankings

Please let me know if you have any questions!
```

## 🔄 Step 6: Updates and Maintenance

### Automatic Deployments
- Any changes pushed to your GitHub repository will automatically redeploy
- App restarts within 1-2 minutes of code changes

### Monitoring
- Check app health via Streamlit Cloud dashboard
- Monitor usage and performance metrics
- View error logs if issues occur

## ⚠️ Important Notes

### Limitations of Free Tier
- **Resource limits**: 1 GB memory, limited CPU
- **Usage limits**: No strict limits but fair usage expected
- **Sleep mode**: App sleeps after inactivity (takes ~30 seconds to wake up)

### Performance Tips
1. **Caching**: All models are cached with `@st.cache_resource`
2. **Batch processing**: Optimize API calls for better performance
3. **Error handling**: Graceful degradation when services are unavailable

## 🆘 Troubleshooting

### Common Issues

**1. App won't start**
- Check secrets configuration
- Verify all API keys are valid
- Check GitHub repository permissions

**2. Authentication not working**
- Clear browser cookies
- Check password complexity
- Verify username/password combination

**3. Slow performance**
- Check API rate limits
- Verify network connectivity
- Monitor resource usage

### Getting Help
- Streamlit Community Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: Create issues in your repository
- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)

## 🎉 Success!

Your AI recruitment platform is now live and accessible to your team with secure authentication!

Next steps:
1. Train your team on the platform features
2. Gather feedback for improvements
3. Monitor usage and performance
4. Plan for scaling if needed 