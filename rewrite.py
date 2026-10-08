import os

if os.path.exists('.env.example'):
    with open('.env.example', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the exposed Django secret key
    if 'django-insecure' in content:
        content = content.replace('django-insecure-+jqdjx9srv#f@80l1wex%7m_br2i2ird!u6c&omyfnj4=lr6p-', 'your_secret_key_here')
        with open('.env.example', 'w', encoding='utf-8') as f:
            f.write(content)
