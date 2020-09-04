import urllib.request

#open the file spanner&screwdriver and download the picture
with open('spanner.txt') as f:
    try:
        for i, line in enumerate(f, 1):
            urllib.request.urlretrieve(line, "Image " + str(i) + " .jpg")
            print(str(i) + " Completed")
    except:
        pass

f.close()
print("Done")