# Todo's

### Article Spreadsheet:
    - [ ] Scrape the [website](https://help.zipboard.co/) into the spreadsheet
        - [ ] Find which LLM to use
            - [ ] Figure out the prompt
        - [ ] How do I make it automated on page refresh (do this later, first get the pages onto the system)
        - [ ] I need to crawl every page while checking:
            - for:
                - Sub-pages
                - Screenshots if they exist or not
                - YT-links
                - Embedded videos
            - if they do:
                - [ ] go till the end, and make a sitemap
                    - can save this to be used as a "has anything been updated" / "do I need to run the job rn ?"
                        - make it so that updates only what has been updated, and does NOT update everything all at once
                - [ ] use that to help populate the basic-barebones CSV
    
    - [ ] LLM time:
        - [ ] Use to populate the empty areas:
            - of:
                - Topics Covered:
                    - Ask the LLM to read the page and tell me the key topics
                        - Or make it simpler and make a smart word counter (very barebones)
                        - Or use a transformer for it to be sentence and context aware and yield the keywords (a little more interesting)
                        - Or use a library (if it exists) (very lazy, but the fastest route)
                    - can also use the name of the reccomended articles and embedded links to help (in the context that will be forwarded to the LLM or whatever)
                
                - Content Type: 
                    - Examples:
                        - "How-to Guide" / "FAQ" / "Troubleshooting"
                    - Can use the "Parent Category" to help with context
                        - Or feed to content of the article to the AI and ask it to give the type of content, given these
                            - If you're feeling VERY lucky, use a pre-trained model to do that (too much effort, don't do that unless you have a lot of time left)
                
                - Gaps Identified (Suggested article outlines for top 2 gaps):
                    - Example:
                        - Missing: Error handling for failed syncs
                    - Can use the "related articles section" to help with this
                    - (OPTIONAL: Because what I am doing is MUCH better) Try to compare to similar tools and see what they're doing better in their documentation
                    - Need a list of everything, basically a big web (THIS IS WHERE THE SITEMAP COMES INTO HELP)
                        - Using this list, I can help plot out connection points where they do not exist, example:
                            - if I have article A linking to article B, I'd say that a "joint" exists
                            - BUT, if it does not, I can use the names of the two articles and generate a article title that would bridge that gap
                            - Use that to fuel article outline for top 2
                        - Export sitemap as a web - with/without joints (with/without gaps) 
                            - Basically a spatial diffusion space
                            - Can use this web to understand the severity (/priority) of the said gap
                                - Using the space / distance between the gap, and somehow add something that checks for relevance of the two topics
                    
                    - All the data to fill out:
                        - GAP ID, GAP desc (by AI & Me), Category, Suggested Article Title, Priority, Rationale
                        - Fields:
                            - Gap ID:
                                - Make a simple iterator that loops over 5 times (or just do it myself)
                            - Gap Description (by AI)
                                - Ask the AI to compare the original articles' gap, and given the new article to be "birthed", what should be it's description in a concise sentence
                            - Gap Description (by me)
                                - Write down what I think the article can help with, given the tool
                            - Category - "Integrations"
                                - Same as the one way above, try to use the method from "CONTENT TYPE"
                            - Suggested Article Title
                                - Simple, create top x AI suggestions that are then ranked, and the best one is submitted
                            - Priority - "High / Medium / Low"
                                - Linked to the map/web
                            - Rationale - aka the reasoning behind why this needs to be included 
                                - Linked to priority

    - [ ] Workflow documentation
        - [ ] Make a diagram with arrows and stuff
        - [ ] Clean up the documentation and folders and naming and all that

### Cleaning up work:

- What is done: 
    - site scraping
    - turning it into a network 
    - tf idf embeddings
    - data cleaning 
    - gap identification v1
    - fixing scraping 

- What's left: 
    - [x] gaps identification v2 to make sure that the gaps are genuine and logical
        - using the content of the articles, make a gap identifier based on the keywords, use the newer table to help with that 

    - [x] based on the topics, need to generate a title 
    - [x] based on the content, need to generate an outline (for 2 gaps)
    - [ ] need to make one table by AI and one by my self
    - [ ] need to format final deliverable spreadsheet

    - [ ] workflow diagram 
    - [ ] LLM prompts

### Future Improvements
- Use an LLM to measure the gap between two articles based on their content (based on the keywords)
    - Not doing this because it's very time consuming
- Use the LLM to generate an article given all the articles and ask it to generate an article which joins two articles which may not be linked in theory