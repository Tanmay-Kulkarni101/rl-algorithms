class DataLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_handle = open(self.file_path, 'w')
        self.data = None

    def write_data(self):
        for line in self.data:
            self.file_handle.write(line)

    def save_data(self):
        pass

class TabularLogger(DataLogger):
    def __init__(self, file_path, headers):
        
        self.parent = super().__init__(file_path)

        self.headers = headers
        self.data = None

    def save_data(self, data):
        if self.data is None:
            self.data = []
            
            if self.headers is not None:
                headers = '\t'.join(self.headers)
            
                self.data.append(headers)
        
        else:
            if isinstance(data, list):
                data = '\t'.join(data)
            elif isinstance(data, dict):
                data_list = []

                for header in self.headers:
                    data_list.append(data[header])
                
                data = data_list 
            else:
                raise NotImplementedError
        
        self.data.append(data)