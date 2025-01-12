
README
==================================
to build
```
docker build -t axisym .
```

to run 

```
docker run -v "${PWD}\test:/code/shared" -p 80:80  axisym   
```