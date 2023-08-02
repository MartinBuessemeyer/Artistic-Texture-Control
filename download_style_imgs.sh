
pushd experiments/target/popular_styles || exit

wget -c https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/rain-princess-cropped.jpg
wget -c https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/3213px-Tsunami_by_hokusai_19th_century.jpg -O the_wave.jpg
wget -c https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/udnie.jpg
wget -c https://img.huffingtonpost.com/asset/5bb235491f000039012379d6.jpeg -O the_shipwreck_of_the_minotaur.jpg
wget -c https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg -O starry_night.jpg
wget -c https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/mosaic.jpg
wget -c https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/candy.jpg
wget -c https://upload.wikimedia.org/wikipedia/en/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg -O femme_nue_anisse.jpg
wget -c https://upload.wikimedia.org/wikipedia/commons/c/c9/Robert_Delaunay%2C_1906%2C_Portrait_de_Metzinger%2C_oil_on_canvas%2C_55_x_43_cm%2C_DSC08255.jpg -O delaunay.jpg
wget -c https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg -O the_scream.jpg

popd || exit