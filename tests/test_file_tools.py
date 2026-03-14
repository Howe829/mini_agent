from tools.file import ReadFileTool, WriteFileTool


def test_read_file_tool():
    read_file_tool = ReadFileTool()
    func_args = r'{"path": "main.py"}'
    result = read_file_tool.call(func_args)
    assert result.is_error is False


def test_write_file_tool():
    write_file_tool = WriteFileTool()
    func_args = r'{"path": "/tmp/temp_file.txt", "content": "Hello, world!\n"}'
    result = write_file_tool.call(func_args)
    assert result.is_error is False
