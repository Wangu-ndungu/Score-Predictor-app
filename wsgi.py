# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 09:41:01 2022

@author: user
"""

from app.main import app
import os 

port = int(os.environ.get("PORT", 5000))
if __name__ == "__main__":
        app.run(host='0.0.0.0', port=port, debug=True)
