## Data Structure for Using Your Own Dataset

All of our datasets follow the structure below. Please prepare any new dataset as follows to use the current dataloader.

If you want to use a different file organization, you will need to write your own dataloader.

```
\double_pendulum
    \0
        \0.png
        \1.png
        \...
    \1
        \...

    \2
        \...
    ...
```