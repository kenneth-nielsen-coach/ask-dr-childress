@echo off
echo Updating YouTube transcripts...
python childress_transcripts.py

echo Updating blog posts...
python childress_blog_scraper.py

echo Updating Substack posts...
python childress_substack_scraper.py

echo Updating papers posted...
python childress_papers.py

echo Updating pdf converters...
python childress_pdf_converters.py

echo Updating pages...
python convert_page_pdfs.py

echo Done! Now commit and push in GitHub Desktop.
pause