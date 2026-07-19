@echo off
echo Updating YouTube transcripts...
python childress_transcripts.py

echo Updating blog posts...
python childress_blog_scraper.py

echo Updating Substack posts...
python childress_substack_scraper.py

echo Done! Now commit and push in GitHub Desktop.
pause