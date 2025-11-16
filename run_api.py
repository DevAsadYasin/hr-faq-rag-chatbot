import os
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent
    os.environ["PYTHONPATH"] = str(project_root)
    
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

