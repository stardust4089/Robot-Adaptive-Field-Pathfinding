# Robot-Adaptive-Field-Pathfinding
### The Model
Ai image recognition is getting better by the day, and the worst it will ever be is however good it is today. FRC has a great opportunity to utilize this developing area of CV programming, espcially with the ease of implementing it into a limelight 3 or by running one through a google Coral TPU.
We already commonly use CV to recognize game piecies, april tags, refelctive tape, and so on. But what about robots? Over the summer of 2023 I went through a hard drive full of robot POV match footage and manually annotated the footage, whcih was then used to train a TFLite model. This model now can recognize robot bumpers with firly high accuracy. 
Some examples of video feeds fed through the model.
The model running on a limelight 3

### So what can we do with this?
Now that I can recognize robots, wouldn;t it be so cool if we could self-drive around them? This is clearly very abitious, but I am partly doing this project as a way to learn. And the more abitious I get, the more mistakes I will make, so the more I will learn. The core of this will act simply as any other coporcessor - a rasp pi that communicates to the Rio through network tables. Into that coprocessor we feed the orientation and location of the robot of the field, and if we also give it the camera's position and FOV during setup, we can do a little math the internet helped me figure out to take the bounding box around the bumper and project it into 3D space. We can then pathfind around that box when doing our pathplanner trajectories to a location, and send that trajectory file back to the rio for use.
Of this I really only have the model and the pathfinding, as well as an interface to configure permanent obstacles so that the robot doesnt go where it can't. I'm currently working on getting the natwork tables interface set up and then building this for a working rasp pi + coral app that can run similar to a limelight (or more accurately, like a glowworm), but I've had to take a hiatus from this project to work on paying for college.