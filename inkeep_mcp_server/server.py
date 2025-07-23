import logging
import typing as T

import openai
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from inkeep_mcp_server.settings import inkeep_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("inkeep-mcp-server")


# https://docs.anthropic.com/en/docs/build-with-claude/citations#plain-text-documents
class InkeepRAGDocument(BaseModel):
    # anthropic fields citation types
    type: str
    source: T.Dict
    title: T.Optional[str] = None
    context: T.Optional[str] = None
    # inkeep specific fields
    record_type: T.Optional[str] = None
    url: T.Optional[str] = None


class InkeepRAGResponse(BaseModel):
    content: T.List[InkeepRAGDocument] = []


async def make_inkeep_rag_request(query: str) -> InkeepRAGResponse:
    logger.info(f"Making Inkeep RAG request with query: {query}")

    async with openai.AsyncOpenAI(
        base_url=inkeep_settings.INKEEP_API_BASE_URL,
        api_key=inkeep_settings.INKEEP_API_KEY,
    ) as openai_client:
        try:
            # https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat
            inkeep_rag_response = await openai_client.beta.chat.completions.parse(
                model=inkeep_settings.INKEEP_API_MODEL,
                messages=[
                    {"role": "user", "content": query},
                ],
                response_format=InkeepRAGResponse,
            )

            inkeep_rag_response_parsed = inkeep_rag_response.choices[0].message.parsed
            if inkeep_rag_response_parsed:
                logger.info(f"Received {len(inkeep_rag_response_parsed.content)} documents in response")
                logger.debug(f"Response content: {inkeep_rag_response_parsed}")
                return inkeep_rag_response_parsed
            else:
                logger.warning("Received empty response from Inkeep API")
                return InkeepRAGResponse()
        except Exception as e:
            logger.error(f"Error making Inkeep RAG request: {str(e)}")
            raise


@mcp.tool(
    name=inkeep_settings.INKEEP_MCP_TOOL_NAME,
    description=inkeep_settings.INKEEP_MCP_TOOL_DESCRIPTION,
)
async def retrieve_product_docs(query: str) -> InkeepRAGResponse:
    logger.info(f"Tool '{inkeep_settings.INKEEP_MCP_TOOL_NAME}' called with query: {query}")
    result = await make_inkeep_rag_request(query)
    logger.info(f"Tool returning {len(result.content)} documents")
    return result
