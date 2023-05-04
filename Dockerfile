FROM rayproject/ray:2e4eb1-py38-cu118

# COPY entrypoint.sh /home/ray/anaconda3/bin/entrypoint.sh

# RUN chmod u+x /home/ray/anaconda3/bin/entrypoint.sh

# ENTRYPOINT ["#!/bin/bash","/home/ray/anaconda3/bin/entrypoint.sh"]
CMD ["tail","-f" ,"/dev/null"]
