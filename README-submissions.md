# Submission instructions and FAQ: 

For the most up-to-date instructions, see the [competition website](https://mila-iqia.github.io/climate-cooperation-competition).



## Registration


Please fill out the [registration form](https://docs.google.com/forms/d/e/1FAIpQLSe2SWnhJaRpjcCa3idq7zIFubRoH0pATLOP7c1Y0kMXOV6U4w/viewform) in order to register for the competition. 

You will only need to provide an email address and a team name. You will also need to be willing to open-source your code after the competition.

After you submit your registration form, we will register it internally. You will need your team name in order to make submissions towards the competition.



## Where can I submit my solution?


*NOTE: Please register for the competition (see the steps above), if you have not done so. Your team must be registered before you can submit your solutions.*

The AI climate competition features 3 tracks.

In Track 1, you will propose and implement multilateral agreements to augment the simulator, and train the AI agents in the simulator. We evaluate the learned policies and resulting economic and climate change metrics.

- The submission form for Track 1 is [here](https://docs.google.com/forms/d/e/1FAIpQLSdATpPMnhjXNFAnGNRU2kuufwD5HFilGxgIXFK9QKsqrDbkog/viewform).

Please select your registered team name from the drop-down menu, and upload a zip file containing the submission files - we will be providing scripts to help you create the zip file.

In Track 2, you will argue why your solution is practically relevant and usable in the real world. We expect the entries in this track to contain a high-level summary for policymakers

- The submission form for Track 2 is [here](https://docs.google.com/forms/d/e/1FAIpQLSeoc4oLBU4c8EoumkocSyhRaxGW0JoEVcBgeuo-U9fSfNOyrQ/viewform).


In Track 3, we invite you to point out potential simulation loopholes and improvements.

- The submission form for Track 3 is [here](https://docs.google.com/forms/d/e/1FAIpQLSed0seSYt8LKywVrE7BARxAPPsO6WYmPUMeIezD7FTV176QvQ/viewform).

If you do not see your team name in the drop-down menu, please contact us on Slack or by e-mail, and we will resolve that for you.




## Scripts for creating the zipped submission file

As mentioned above, the zipped file required for submission is automatically created post-training. However, for any reason (for example, for providing a trained policy model at a different timestep), you can create the zipped submission yourself using the `create_submizzion_zip.py` script. Accordingly, create a new directory (say `submission_dir`) with all the relevant files (see the section above), and you can then simply invoke
```commandline
python scripts/create_submission_zip.py -r <PATH-TO-SUBMISSION-DIR>
```

That will first validate that the submission directory contains all the required files, and then provide you a zipped file that can you use towards your submission.


## Scripts for unit testing

In order to make sure that all the submissions are consistent in that they comply within the rules of the competition, we have also added unit tests. These are automatically run also when the evaluation is performed. The script currently performs the following tests

- Test that the environment attributes (such as the RICE and DICE constants, the simulation period and the number of regions) are consistent with the base environment class that we also provide.
- Test that the `climate_and_economy_simulation_step()` is consistent with the base class. As aforementioned, users are free to add different negotiation strategies such as multi-lateral negotiations or climate clubs, but should not modify the equations underlying the climate and economic dynamics in the world.
- Test that the environment resetting and stepping yield outputs in the desired format (for instance, observations are a dictionary keyed by region id, and so are rewards.)
- If the user used WarpDrive, we also perform consistency checks to verify that the CUDA implementation of the rice environment is consistent with the pythonic version.

USAGE: You may invoke the unit tests on a submission file via
```commandline
python scripts/run_unittests.py -r <PATH-TO-ZIP-FILE>
```


## Scripts for performance evaluation

Before you actually upload your submission files, you can also evaluate and score your submission on your end using this script. The evaluation script essentially validates the submission files, performs unit testing and computes the metrics for evaluation. To compute the metrics, we first instantiate a trainer, load the policy model with the saved parameters, and then generate several episode rollouts to measure the impact of the policy on the environment.

USAGE: You may evaluate the submission file using
```commandline
python scripts/evaluate_submission.py -r <PATH-TO-ZIP-FILE>
```
Please verify that you can indeed evaluate your submission, before actually uploading it.


# Evaluation process

After you submit your solution, we will be using the same evaluation script that is provided to you, to score your submissions, but using several rollout episodes to average the metrics such as the average rewards, the global temperature rise, capital, production, and many more. We will then rank the submissions based on the various metrics.The score computed by the evaluation process should be similar to the score computed on your end, since they use the same scripts.


## What happens when I make an invalid submission?

An "invalid submission" may refer to a submission wherein some or all of the submission files are missing, or the submission files are inconsistent with the base version, basically anything that fails in the evaluation process. Any invalid solution cannot be evaluated, and hence will not feature in the leaderboard. While we can let you know if your submission is invalid, the process is not automated, so we may not be able to do it promptly. To avoid any issues, please use the `create_submission_zip` script to create your zipped submission file.


# Leaderboard

The competition leaderboard is displayed on the [competition website](https://mila-iqia.github.io/climate-cooperation-competition). After you submit your valid submission, please give it a few minutes to perform an evaluation of your submission and refresh the leaderboard.


# How many submissions are allowed per team?

There is no limit on the number of submissions per team. Feel free to submit as many solutions as you would like. We will only be using your submission with the highest evaluation score towards the leaderboard.