# Free-response experiment

This experiment tests code-switching behavior in a free-response format.

## generating videos
NOTE: manim has a bug where it caches images improperly causing some of the avatars to flicker within the same video.

To regenerate the problematic videos, run the following command:

```
python stim/scripts/regenerate_problematic_files.py
```


## scp ing to MIT scripts

`cd experiments/3pp`

`rsync -av --exclude 'stim/convo_vids/videos/480p15/partial_movie_files' free-response/ aliciach@athena.dialup.mit.edu:~/www/tangrams/free-response`

website is at
https://web.mit.edu/aliciach/www/tangrams/free-response