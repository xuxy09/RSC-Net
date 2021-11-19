1. Install pyrender
```
pip install pyrender
```

2. Install osmesa
```
sudo apt update
sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -f
```

3. Install a compatible pyopengl
```
git clone https://github.com/mmatl/pyopengl.git
pip install ./pyopengl
```

4. Remember to set the environment as osmesa. Put this at the beginning of your code:
```
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
```

---

You should be able to work with pyrender now.
The above instructions are from https://pyrender.readthedocs.io/en/latest/install/index.html#getting-pyrender-working-with-osmesa
