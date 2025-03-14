import logging
import os

if not os.path.exists('log'): os.makedirs('log')

def return_logger(loggerName: str = 'info', outputFile: str = 'info.log', stream: bool = False, level: str = 'info', overwrite: bool = False):
    '''
    return logger
    Args:
        loggerName: logger name
        outputFile: output file name
        stream: whether to output to stream
        level: log level
        overwrite: whether to overwrite existing log file
    '''
    logger = logging.getLogger(loggerName)
    # Clear existing handlers if overwrite is True
    if overwrite:
        logger.handlers.clear()
        
    if outputFile is not None and outputFile:
        mode = 'w' if overwrite else 'a'
        log_file_handler = logging.FileHandler(f'./log/{outputFile}', mode=mode, delay=False)
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(log_file_handler)
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)
    assert level in ['debug', 'info', 'warning', 'error', 'critical']
    str2level = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'critical': logging.CRITICAL}
    logger.setLevel(str2level[level])
    
    return logger

# info_logger = return_logger('info', 'info.log', False, 'error')
# bug_logger = return_logger('bug', 'bug.log', False, 'error')