import pytest
import pandas as pd
from main import load_knowledge_base, get_relevant_knowledge
from sklearn.feature_extraction.text import TfidfVectorizer

def test_knowledge_base_loading(monkeypatch):
    # Mock CSV data
    mock_df = pd.DataFrame({
        'question': ['How to deal with anxiety?'],
        'answer': ['Practice deep breathing...']
    })
    monkeypatch.setattr('pandas.read_csv', lambda x: mock_df)
    
    kb = load_knowledge_base()
    assert 'df' in kb
    assert 'vectorizer' in kb
    assert 'question_vectors' in kb

def test_relevant_knowledge_retrieval():
    # Mock knowledge base
    mock_df = pd.DataFrame({
        'question': ['How to deal with anxiety?'],
        'answer': ['Practice deep breathing...']
    })
    kb = {
        'df': mock_df,
        'vectorizer': TfidfVectorizer().fit(mock_df['question']),
        'question_vectors': TfidfVectorizer().fit_transform(mock_df['question'])
    }
    
    entries = get_relevant_knowledge("I'm feeling anxious", kb)
    assert len(entries) > 0