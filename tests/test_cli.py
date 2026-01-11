import pytest
from unittest.mock import patch, MagicMock
from csttool.cli import main

def test_cli_version(capsys):
    """Test that --version flag works."""
    with patch('sys.argv', ['csttool', '--version']):
        # expect SystemExit
        with pytest.raises(SystemExit):
            main()
    
    captured = capsys.readouterr()
    assert "csttool" in captured.out or "csttool" in captured.err

def test_cli_help(capsys):
    """Test that --help flag works."""
    with patch('sys.argv', ['csttool', '--help']):
        with pytest.raises(SystemExit):
            main()
            
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "usage:" in captured.err

@patch('csttool.cli.cmd_check')
def test_cli_check_command(mock_cmd_check):
    """Test that 'check' command calls the correct function."""
    with patch('sys.argv', ['csttool', 'check']):
        main()
        mock_cmd_check.assert_called_once()    
