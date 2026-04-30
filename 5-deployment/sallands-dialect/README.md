# De Sallandse woordengenerator

## Summary
To run ‘de Sallandse taalgenerator’ execute the following steps:
-	Start up the Surf VM → UOS3JGerrits7
-	Run `--docker compose up -d`
-	Go to: http://145.38.204.202/ 



## Short reflection - Deployment portfolio assignment

First the straattaal folder was cloned. The model was trained on the ‘Sallands’ dialect dataset through scraping: https://www.mijnwoordenboek.nl/dialect/Sallands . This dialect was selected because it was used by my grandparents, which was always funny to listen to. ‘Goedgoan!’

After testing and checking the files of the straattaal package, the following files were created:
- Dockerfile
- .dockerignore
- MakeFile (steps are only executed if the required files are missing)
- Docker compose file (.yml)

During this process the following errors were observed and fixed:
- Testing on a local machine does not allow for port 80, therefore port 8000 was used on a local machine.
- The path with ‘static’ in the FileResponse seems to be giving internal server errors. After troubleshooting with ChatGPT the path was changed from `-- return FileResponse("static/index.html")` to `--return FileResponse(FRONTEND_FOLDER / "index.html")`.
- The torch-python-slim image was changed from ARM64 to AMD64, (both images include uv).
- FastAPI was configured as an optional dependency. This gave an error and therefore FastAPI was moved to the main dependencies. This increased the size of the installation, but as it is necessary for front end deployment, this was assumed to be necessary.
- Some stupid mistakes due to inexperience occurred with the docker compose file. It needs to be stored as docker-compose.yml instead of dockercompose.yml.
- The HTML file was tweaked to the new application. 

