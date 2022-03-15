# reddit_posts.py                  By: Samer Al-khateeb
# This is a modified version of the code provided here 
# https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
# given a subreddit name all its posts will be collected


# go to main() function and enter the needed credentials
# and the subreddit name then run the code, you should get 
# an output file named reddit_posts_output.csv

import requests
import json
import csv
from datetime import datetime


def write_output_to_CSV(biglist):
    # creating a file to save the output
    columnNames =["subreddit", "subredditSubscribers", "subredditType", 
                        "postAuthor", "postID", "postTitle", 
                        "postText", "postDateUTC", "postURL", "postType", 
                        "postNumComments", "postNumCrossPosts", "postUpvoteRatio", 
                        "postUps", "postDowns", "postScore"]
    with open('reddit_posts_output.csv', 'w') as csvOutputFile:
        #creating a csv writer object 
        csvwriter = csv.writer(csvOutputFile)
        #write the columns header
        csvwriter.writerow(columnNames)
        #writing/inserting the list to the output file 
        csvwriter.writerows(biglist)


def main():
        ########## Enter Info in this Section of the Script ##################

        # Enter your Reddit account credentials here
        username = 'PasteYoursHere!'
        password = 'PasteYoursHere'

        # Enter your APP credentials here
        applicationName = 'PasteYoursHere'
        applicationID = 'PasteYoursHere'
        applicationSecret = 'PasteYoursHere'

        # enter the subreddit name here
        subreddit = 'PasteYoursHere'

        # which posts you want to retrive? 
        # for most popular --> set to 'hot'
        # for most recent --> set to 'new'
        whichPosts = 'hot'

        # how many requests you want to make?
        numOfRequests = 3

        # number of results per request?
        numOfResults = 100
        
        ########### No info needed below this line of code #################
        
        # creating an authorization
        auth = requests.auth.HTTPBasicAuth(applicationID, applicationSecret)

        # here we pass our login method (password), username, and password
        data = {'grant_type': 'password',
                'username': username,
                'password': password}
        
        # setup our header info, which gives reddit a brief description of our app
        headers = {'User-Agent': '{}/0.0.1'.format(applicationName)}

        # send our request for an OAuth token
        res = requests.post('https://www.reddit.com/api/v1/access_token',
                        auth=auth, data=data, headers=headers)

        # convert response to JSON and pull access_token value
        TOKEN = res.json()['access_token']

        # add authorization to our headers dictionary
        headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

        # while the token is valid (~2 hours) we just add headers=headers to our requests
        requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)
        
        # initialize parameters for pulling data in loop
        params = {'limit': numOfResults}

        # creating a list to hold the output
        CSV_output_list =[]

        # if numOfRequests is set to 10 and numOfResults set to 100. It will return 1000 posts
        for numOfRequests in range(numOfRequests):
                # making a request to get data
                res = requests.get("https://oauth.reddit.com/r/{}/{}".format(subreddit,whichPosts), headers=headers, params=params)

                # convert the response into JSON
                jsonResponse = res.json()

                # uncomment the line below if you want to see the JSON response on the screen
                #print(json.dumps(jsonResponse, indent=4, sort_keys=True))
                
                # for each of the 100 posts, extract it's data
                for post in jsonResponse['data']['children']:
                        subreddit = post['data']['subreddit']
                        subredditSubscribers = post['data']['subreddit_subscribers']
                        subredditType = post['data']['subreddit_type']
                        
                        postAuthor = post['data']['author']
                        postID = post['data']['id']
                        postTitle = post['data']['title']
                        postText = post['data']['selftext']
                        postDateUTC = datetime.fromtimestamp(post['data']['created_utc']).strftime('%Y-%m-%dT%H:%M:%SZ')
                        postURL = post['data']['url']
                        
                        try:
                            postType = post['data']['post_hint']
                        except KeyError as e:
                            postType = "undefined!"


                        postNumComments = post['data']['num_comments']
                        postNumCrossPosts = post['data']['num_crossposts']
                        postUpvoteRatio = post['data']['upvote_ratio']
                        postUps = post['data']['ups']
                        postDowns = post['data']['downs']
                        postScore = post['data']['score']
                        
                        # create a CSV row
                        CSV_output_row = [subreddit, subredditSubscribers, subredditType, 
                                                postAuthor, postID, postTitle, 
                                                postText, postDateUTC, postURL, postType,
                                                postNumComments, postNumCrossPosts, postUpvoteRatio, 
                                                postUps, postDowns, postScore]
                        
                        # append the row to the CSV list
                        CSV_output_list.append(CSV_output_row)
                
                # determine the last post name, so we can make another request to the other 100 posts
                responseLength = len(jsonResponse['data']['children'])
                try:
                    nextName = jsonResponse['data']['children'][responseLength-1]['data']['name']
                    params['after'] = nextName
                except IndexError as e: 
                    break
                    
        # send the list to the function to create a CSV file
        write_output_to_CSV(CSV_output_list)

        print()
        print("The script finished and an output file should be generated!") 
        print() 

if __name__ == "__main__":
    main()