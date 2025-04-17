import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.seed_user import create_profile

class TestProfileCreator:
    
    @pytest.fixture
    def mock_recommender(self):
        recommender = MagicMock()
        recommender.reset_preferences.return_value = True
        recommender.like_title.return_value = True
        recommender.dislike_title.return_value = True
        return recommender
    
    @pytest.fixture
    def mock_failing_recommender(self):
        recommender = MagicMock()
        recommender.reset_preferences.return_value = True
        
        # Set up like_title and dislike title to fail for specific titles
        def mock_like_title(title):
            if title in ["NOVA: Black Hole Apocalypse", "Paw Patrol"]:
                return False
            elif title == "Octonauts":
                raise Exception("Test exception")
            return True
            
        def mock_dislike_title(title):
            if title in ["Narcos", "Ozark"]:
                return False
            elif title == "Pulp Fiction":
                raise Exception("Test exception")
            return True
            
        recommender.like_title.side_effect = mock_like_title
        recommender.dislike_title.side_effect = mock_dislike_title
        return recommender
    
    def test_create_profile_successful(self, mock_recommender, capsys):
        result = create_profile(mock_recommender)
        
        mock_recommender.reset_preferences.assert_called_once()
        
        # Chheck expected total of likes and dislikes
        expected_likes = 16 
        expected_dislikes = 26 
        
        assert mock_recommender.like_title.call_count == expected_likes
        
        assert mock_recommender.dislike_title.call_count == expected_dislikes
        
        # Check the output messages
        captured = capsys.readouterr()
        assert f"Successfully liked {expected_likes} titles and disliked {expected_dislikes} titles" in captured.out
        
        assert result == mock_recommender
    
    def test_create_profile_with_failures(self, mock_failing_recommender, capsys):
        result = create_profile(mock_failing_recommender)
        
        mock_failing_recommender.reset_preferences.assert_called_once()
        
        # Check expxected total of successful likes and dislikes
        expected_successful_likes = 13  
        expected_successful_dislikes = 23  
        
        # Check the output messages for errors
        captured = capsys.readouterr()
        
        assert "Failed to like: NOVA: Black Hole Apocalypse" in captured.out
        assert "Failed to like: Paw Patrol" in captured.out
        assert "Error liking title Octonauts: Test exception" in captured.out
        
        assert "Failed to dislike: Narcos" in captured.out
        assert "Failed to dislike: Ozark" in captured.out
        assert "Error disliking title Pulp Fiction: Test exception" in captured.out
        
        assert f"Successfully liked {expected_successful_likes} titles and disliked {expected_successful_dislikes} titles" in captured.out
        
        # Check that the function returns the recommender object
        assert result == mock_failing_recommender
    
    @patch('builtins.print')
    def test_create_profile_print_messages(self, mock_print, mock_recommender):
        create_profile(mock_recommender)
        
        # Check that we printed messages for the below

        mock_print.assert_any_call("Successfully liked 16 titles and disliked 26 titles")
        
        mock_print.assert_any_call("Liked: Our Planet")
        mock_print.assert_any_call("Liked: Planet Earth II")
        
        mock_print.assert_any_call("Disliked: Narcos")
        mock_print.assert_any_call("Disliked: Orange Is the New Black")