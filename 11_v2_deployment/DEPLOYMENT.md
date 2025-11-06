# 🚀 Kubernetes Copilot - Vercel Deployment Guide

This guide explains how to deploy the Kubernetes Copilot application to Vercel.

## 📋 Prerequisites

- [Vercel Account](https://vercel.com)
- [GitHub Repository](https://github.com) (for automatic deployments)
- OpenAI API Key

## 🏗️ Project Structure

The application has been converted from Streamlit to a Vercel-compatible architecture:

```
11_v2_deployment/
├── api/
│   └── main.py              # FastAPI backend (Vercel serverless function)
├── public/
│   ├── index.html           # Frontend HTML
│   ├── styles.css           # CSS styles
│   └── app.js               # JavaScript frontend logic
├── k8s_copilot/             # Core application logic
│   ├── agents/              # LangGraph agents
│   ├── data/                # Mock K8s data
│   ├── tools/               # K8s tools
│   ├── vector_db/           # Vector database utilities
│   └── ...
├── vercel.json              # Vercel configuration
├── requirements.txt         # Python dependencies (optimized)
└── README.md                # Updated documentation
```

## 🔧 Configuration Changes

### Backend Changes (Lightweight Version)
- **FastAPI**: Replaced Streamlit with FastAPI for serverless compatibility  
- **Ultra-Lightweight**: Minimal dependencies (<250MB) to meet Vercel limits
- **OpenAI Integration**: Direct GPT-4o-mini integration without heavy agent frameworks
- **Mock Data**: Uses structured mock data instead of vector database for demonstrations
- **Simplified Agents**: Basic OpenAI chat instead of full LangGraph orchestration

### Frontend Changes
- **Web-based UI**: Pure HTML/CSS/JavaScript instead of Streamlit
- **API Communication**: RESTful API calls to backend
- **Responsive Design**: Mobile-friendly interface
- **Interactive Charts**: Chart.js for cost visualization

## 🚀 Deployment Steps

### 1. Prepare Your Repository

1. **Push to GitHub:**
   ```bash
   cd 11_v2_deployment
   git init
   git add .
   git commit -m "Initial Vercel deployment setup"
   git remote add origin https://github.com/YOUR_USERNAME/k8s-copilot-vercel.git
   git push -u origin main
   ```

### 2. Deploy to Vercel

#### Option A: Vercel Dashboard (Recommended)

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will automatically detect the configuration from `vercel.json`
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key (required)
   - `COHERE_API_KEY`: Your Cohere API key (optional, but recommended for enhanced retrieval)
6. Click "Deploy"

#### Option B: Vercel CLI

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Login and deploy:**
   ```bash
   vercel login
   vercel
   ```

3. **Set environment variables:**
   ```bash
   vercel env add OPENAI_API_KEY
   # Enter your OpenAI API key when prompted
   
   vercel env add COHERE_API_KEY
   # Enter your Cohere API key when prompted (optional but recommended)
   ```

### 3. Configure Environment Variables

In your Vercel dashboard, go to your project settings and add:

- **OPENAI_API_KEY**: Your OpenAI API key (required)
- **COHERE_API_KEY**: Your Cohere API key (optional, for enhanced retrieval)

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for language model access |
| `COHERE_API_KEY` | Recommended | Cohere API key for enhanced retrieval methods and reranking |

## 📊 Features

The deployed application includes:

### 💬 Chat Interface
- Natural language queries about Kubernetes clusters
- Two agent types: Copilot Agent (with tools) and RAG Agent (simple Q&A)
- Real-time responses with error handling

### 💰 Cost Analysis
- Interactive cost charts and visualizations
- Deployment cost breakdowns
- Cost optimization suggestions

### 📊 Resource Dashboard
- Resource utilization metrics
- Cluster overview statistics
- Performance monitoring

## 🧪 Testing Your Deployment

1. **Access your deployed app** at `https://your-project-name.vercel.app`
2. **Test the API endpoint** at `https://your-project-name.vercel.app/api/health`
3. **Try example queries:**
   - "What are the costs of my Kubernetes deployments?"
   - "How many GPUs does the ml-training deployment use?"
   - "Show me optimization opportunities"

## 🐛 Troubleshooting

### Common Issues

1. **API Key Not Working**
   - Verify environment variable is set correctly in Vercel dashboard
   - Check API key validity in OpenAI dashboard

2. **Slow Cold Starts**
   - This is normal for serverless functions
   - Consider upgrading to Vercel Pro for better performance

3. **Package Size Issues**
   - Dependencies are optimized for Vercel
   - If you add new packages, keep bundle size under 50MB

4. **CORS Issues**
   - CORS is configured in the FastAPI backend
   - Check browser console for specific errors

### Debug Mode

To enable debug logging, add this environment variable:
- `DEBUG`: Set to `true`

## 🔄 Updates and Maintenance

### Updating the Application

1. **Make changes** to your code
2. **Commit and push** to GitHub:
   ```bash
   git add .
   git commit -m "Your update message"
   git push
   ```
3. **Vercel automatically redeploys** from GitHub

### Monitoring

- **Check deployment logs** in Vercel dashboard
- **Monitor function usage** and errors
- **Set up alerts** for critical issues

## 📈 Performance Considerations

### Serverless Limitations
- **Function timeout**: 30 seconds (configured in vercel.json)
- **Memory limit**: 1GB (Vercel default)
- **Cold starts**: First request may be slower

### Optimization Tips
- Vector store initialization is cached
- Dependencies are minimized
- Static assets are served from Vercel's CDN

## 🔒 Security

- **API keys** are stored as environment variables
- **CORS** is configured for your domain
- **No sensitive data** in client-side code

## 💡 Next Steps

After successful deployment, consider:

1. **Custom domain** setup in Vercel
2. **Analytics** integration
3. **Real Kubernetes cluster** integration
4. **Enhanced monitoring** and alerting

## 🆘 Support

If you encounter issues:

1. **Check Vercel function logs** in the dashboard
2. **Review browser console** for frontend errors
3. **Test API endpoints** directly
4. **Verify environment variables** are set correctly

## 📚 Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)

---

**🎉 Congratulations!** Your Kubernetes Copilot is now deployed on Vercel and ready to help manage your Kubernetes clusters through natural language interactions.
