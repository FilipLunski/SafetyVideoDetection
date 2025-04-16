import cv2
import numpy as np
import urllib.request

class Decoder:
    def __init__(self, ip_address, path = 'control/faststream', params='stream=MxPEG&preview&previewsize=640x480&quality=40&fps=4.0'):
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        
        url = f'http://{ip_address}{path}?{params}'
        top_level_url = f"http://{ip_address}/"
        
        # Add the username and password.
        # If we knew the realm, we could use it instead of None.
        password_mgr.add_password(None, top_level_url, "user", "mobotix")

        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

        # create "opener" (OpenerDirector instance)
        opener = urllib.request.build_opener(handler)

        # use the opener to fetch a URL
        opener.open(url)

        # Install the opener.
        # Now all calls to urllib.request.urlopen use our opener.
        urllib.request.install_opener(opener)


        self.stream = urllib.request.urlopen(url)
    def read(self):
        
        total_bytes = b''
        total_bytes += self.stream.read(1024)
        b = total_bytes.find(b'\xff\xd9') # JPEG end
        if not b == -1:
            a = total_bytes.find(b'\xff\xd8') # JPEG start
            jpg = total_bytes[a:b+2] # actual image
            total_bytes= total_bytes[b+2:] # other informations
            return True, np.fromstring(jpg, dtype=np.uint8)
        return False, None            