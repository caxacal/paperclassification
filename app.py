@app.route('/')
def home():
    """Root endpoint with service info"""
    return jsonify({
        'status': 'active',
        'service': 'PSITE Abstract Classifier API',
        'version': '1.0'
    })

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy'
    })
