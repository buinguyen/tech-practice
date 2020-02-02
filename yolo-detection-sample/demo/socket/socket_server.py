import socket

RTSP_PORT_DEFAULT = "8554"
URL_STR = "/test"

class SocketServer():
    def __init__(self, host, port):
        self.maxclients = 1
        self.maxbytes = 1024
        self.host = host
        self.port = port
        self.i = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, int(self.port)))
        self.sock.listen(self.maxclients)
        print "Socket server is listening: ", self.host, self.port

    def close(self):
        if not self.sock:
            self.sock.close()

    def makeResponse(self, gender):
        url = "rtsp://" + self.host + ":" + RTSP_PORT_DEFAULT + URL_STR
        data = '{"url":"' + url + '",' + '"gender":"' + gender + '"}\n'
        print "Socket send: ", data
        return data

    def communicate(self, gender):
        while True:
            # Accept client
            conn, addr = self.sock.accept()
            print "Accept connection host: ", addr

            index = conn.recv(self.maxbytes)
            print "Received Request: ", index
               
            if not index:
                print "Client is closed!!!"
            else:
                conn.send(self.makeResponse(gender)) 
