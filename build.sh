#!/usr/bin/env bash

# Create model directory
mkdir -p model

echo "Build completed successfully"
```

### 4. `.gitignore`
```
__pycache__/
*.pyc
.env
model/
*.pt
*.pkl