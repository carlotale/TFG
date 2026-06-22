#!/usr/bin/env python

"""Generate JSON schema for the configuration file using Pydantic."""

import json

from llm_conversation.config import Config

# Generate current schema
current_schema = Config.model_json_schema()
# Output schema to stdout
print(json.dumps(current_schema, indent=4))
