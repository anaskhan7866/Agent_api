from django.test import TestCase
from django.urls import reverse
import json

class RAGServiceTestCase(TestCase):
    def test_query_revenue_comparison(self):
        url = reverse('query-rag')
        payload = {
            "query": "Compare the revenue generated from automotive leasing versus energy generation and storage."
        }
        
        response = self.client.post(
            url, 
            data=json.dumps(payload), 
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200, response.json().get('error', 'Unknown Error'))
        data = response.json()
        
        self.assertIn('answer', data)
        self.assertIn('sources', data)
