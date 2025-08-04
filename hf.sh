#!/bin/sh

curl -O https://d36u8deuxga9bo.cloudfront.net/certificates/Cisco_Umbrella_Root_CA.cer
openssl x509 -inform PEM -in Cisco_Umbrella_Root_CA.cer -out cisco.crt
rm -rf Cisco_Umbrella_Root_CA.cer
mv cisco.crt /usr/local/share/ca-certificates/
update-ca-certificates