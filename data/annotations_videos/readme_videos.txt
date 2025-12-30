Video Dataset

Videos:

Our video dataset contains a total of 3,477 advertisement videos from YouTube. We provide the videos as a list of YouTube IDs, in the file final_video_id_list.csv . The videos can be found at https://www.youtube.com/watch?v=[insert ID here]. Please remove single quotes from the video ID before entering it in the URL.

http://people.cs.pitt.edu/~kovashka/ads/final_video_id_list.csv

Annotation Files:

The annotation files are in json format. The key in the annotation file is the video ID. The value can take different forms depending on the type of annotation, and is described below.

We provide two types of annotations:
(1) Raw annotations directly from annotators -- these are of three subtypes:
	(a) free-form annotations that we used in order to define our list of classes for the Topics and Sentiments annotations;
	(b) free-form annotations for QA Action and Reason ("What should I do?" and "Why should I do it?") that we did not process further; 
	(c) multiple-choice selections of *individual* annotators (usually 5 per video), for Topics, Sentiments, Funny, Exciting, Language, and Effective;
(2) Cleaned annotations -- these compute a majority vote over the raw annotations, except for QA annotations. Thus there is a single annotation given for each video. If the raw annotation is in free-form text, we semi-manually map it to one of the multiple-choice options.  

For "Topics" and "Sentiments", for consistency, the raw files contain only strings. However, note that these strings inherently are of two types: (1) strings that annotators wrote free-form, or (2) strings that correspond to the multiple-choice selection that an annotator made (where we use the abbreviation to represent the class). For example:
{
  ...,
  Pzl86IjTpHI: ["media", "award show", "media", "media", "media"],
  h6CcxJQq1x8: ["restaurant", "soda", "soda", "soda", "soda"],
  ...
}

In contrast, the cleaned files show the class ID, and a corresponding mapping is provided in a separate txt file (Topics_List.txt and Sentiments_List.txt, respectively). The majority class ID was computed from the topics/sentiments that annotators selected. 
 
For Q/A ("Action", "Reason"), we provide unprocessed annotations as free-form sentences. The file QA_Action.json contains the results for the "What" questions, and the file QA_Reason.json contains the results for the "Why" questions for the same images. For example:
{
  ...,
  5AuLkMBAFZg: ["Because it could make me sick.",
                "Because what you put in your mouth could be harmful to you.",
                "Because it can be dangerous. ",
                "Because its reminding children to not put things in their mouth and explaining the dangers of it.",
                "Because I could get sick."],
  ...
}

For "Funny" and "Exciting", "1" means funny or exciting, respectively, and "0" means NOT funny/exciting. For the majority vote, we calculate a score ranging from 0 to 1 based on the average of the annotators' responses. For example, if all 5 annotators agree that the video is funny, then it will be 5/5 = 1; if 2 annotators thought it's funny and 3 thought it's not, then the funny score will be 2/5 = 0.4. We used the 'score' as the cleaned annotations for funny/exciting. In our SVM experiments, we choose 0.7 as a minimum threshold for positive (funny/exciting) and 0.3 as a maximum threshold for negative. Thus, videos with score over 0.7 will be treated as positive training samples, and scores below 0.3 will be treated as negative ones. We did not train our classifier on the ambiguous videos.

For "Language", "1" means "English", "0" means "non-English" and "-1" means "language does not matter to understand the ad".

For "Effective", the score ranges from "1" to "5", with "5" being "most effective". 

Citation:

If you use our data, please cite the following paper:

Automatic Understanding of Image and Video Advertisements. Zaeem Hussain, Mingda Zhang, Xiaozhong Zhang, Keren Ye, Christopher Thomas, Zuha Agha, Nathan Ong, Adriana Kovashka. To appear, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.
