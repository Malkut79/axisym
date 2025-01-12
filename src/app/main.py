from typing import Union

from fastapi import FastAPI

# import os
# files = [f for f in os.listdir('.') if os.path.isfile(f)]
# for f in files:
#     print(f)


from app.axisym import disc




app = FastAPI()


@app.get("/")
def read_root():

    x = disc()

    return {"Hello": "World", "message": x}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}