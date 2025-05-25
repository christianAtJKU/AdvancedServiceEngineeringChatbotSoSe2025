# TrustChat
**TrustChat** is a chat interface with bias mitigation, data pseudonymization, and security features.

## Setup
```bash
# create virtual environment
python -m venv .venv

# activate virtual environment
# on Windows:
.venv\Scripts\activate
# on macOS/Linux:
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# set API key for LLM access (replace with actual key)
export GROQ_API_KEY="gsk_…"

# start application
uvicorn server:app
```

## Test

Hi, my name is John Smith (friends call me Johnny).  
I'm a 34-year-old Black Christian gay software engineer from Berlin working at Google.  
My blind wife Mary Johnson studies at Harvard University.  
You can reach me at john.smith@example.com or call me on +4312345678.  
We'll meet the European Council delegation on 12 July 2025 at 3:00 pm.