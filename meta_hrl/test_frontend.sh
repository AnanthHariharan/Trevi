#!/bin/bash

echo "Testing Meta-HRL Frontend Setup..."

# Test if npm dependencies are installed
if [ ! -d "frontend/frontend/node_modules" ]; then
    echo "❌ Node modules not found. Running npm install..."
    cd frontend/frontend && npm install
    cd ../..
fi

# Test if the build works
echo "🔨 Testing production build..."
cd frontend/frontend
npm run build > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Production build successful"
else
    echo "❌ Production build failed"
    exit 1
fi

# Clean up build directory
rm -rf build

echo "🎉 Frontend setup test completed successfully!"
echo ""
echo "To start the development server:"
echo "  cd meta_hrl/frontend/frontend"
echo "  npm start"
echo ""
echo "To start the backend:"
echo "  cd meta_hrl"
echo "  python start_dashboard.py"