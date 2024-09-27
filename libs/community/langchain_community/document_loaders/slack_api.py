import logging
import os
from typing import Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.slack import (
    DEFAULT_CHANNEL_TYPES,
    DEFAULT_MESSAGE_LIMIT,
    SlackAPIWrapper,
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SlackAPILoader(BaseLoader):
    """Load from a `Slack` API."""

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        workspace_url: Optional[str] = None,
        channel_name: Optional[str] = None,
        load_auth: Optional[bool] = False,
        message_limit: Optional[int] = DEFAULT_MESSAGE_LIMIT,
        channel_types: Optional[str] = DEFAULT_CHANNEL_TYPES,
    ):
        """
        Initialize the SlackAPILoader.

        Args:
            token (Optional[str]): The Slack API token.
            workspace_url (Optional[str]): The Slack workspace URL.
                Including the URL will turn sources into links. Defaults to None.
            channel_name (Optional[str]): The channel to load data from.
                When not provided, loads data from all channels. Defaults to None.
            load_auth (Optional[bool]): Whether to load authorized identities.
                Defaults to False.
            channel_types (Optional[str]): Comma-separated list of types of channels to
                load data from. Defaults to "public_channel,private_channel".
            message_limit (Optional[int]): The maximum number of messages to load.
                Defaults to 1000.
        """
        self.token = token
        # Get workspace url from environment variable if not provided
        self.workspace_url = workspace_url or os.getenv("SLACK_WORKSPACE_URL", "")
        self.channel_name = channel_name
        self.load_auth = load_auth
        self.message_limit = message_limit
        self.channel_types = channel_types
        self.client = SlackAPIWrapper(token=token, workspace_url=workspace_url)
        # Get channel details map(channel name to channel details)
        self.channel_details_map = self.client.get_channel_details_map(channel_types)
        # Authorized identities map(channel name to authorized identities)
        self._authorized_identities_map: dict = {}
        if self.load_auth:
            # Get authorized identities if load_auth is True
            self._load_authorized_identities()

    @staticmethod
    def _enriched_message_text(message: dict) -> str:
        """
        Enrich the message text with replies in the thread.

        Args:
            message (dict): The message to enrich.

        Returns:
            str: The enriched message text.
        """
        replies = message.get("replies", [])
        if not replies:
            return message.get("text", "")
        # Get text from each reply(First reply is the original message)
        reply_texts = [reply.get("text", "") for reply in replies]
        return "\n\n".join(reply_texts)

    def _get_channels(self) -> list:
        """
        Get the list of channels to load data from.

        Returns:
            list: A list of channel names.
        """
        return (
            [self.channel_name]
            if self.channel_name
            else list(self.channel_details_map.keys())
        )

    def _load_authorized_identities(self) -> None:
        """
        Load authorized identities for each channel.
        An authorized identity is a user who has access to the channel.
        """

        channels = self._get_channels()
        # Get user details map(user ID to user details)
        self.user_details_map = self.client.get_user_details_map()

        # Get authorized identities for each channel
        for channel_name in channels:
            _authorized_identities = self.client.get_authorized_identities(
                channel_name, self.user_details_map, self.channel_details_map
            )
            self._authorized_identities_map[channel_name] = _authorized_identities

    def lazy_load(self) -> Iterator[Document]:
        """
        Load and return documents from the Slack API.
        If a channel_name is specified, only messages from that channel are loaded.

        Yields:
            Document: A document object representing the parsed blob.
        """
        # Get the list of channels to load data from
        channels = self._get_channels()
        # Get messages from each channel
        for channel_name in channels:
            # Get the channel ID
            channel_id = self.channel_details_map.get(channel_name, {}).get("id")
            if not channel_id:
                logger.warning(f"Channel ID not found for channel: {channel_name}")
                continue
            # Get messages from the channel
            messages = self.client.get_messages(
                channel=channel_id, limit=self.message_limit
            )
            # Skip if no messages found
            if not messages:
                logger.warning(f"No messages found for channel: {channel_name}")
                continue

            for message in messages:
                yield self._convert_message_to_document(message, channel_name)

    def _convert_message_to_document(
        self,
        message: dict,
        channel_name: str,
    ) -> Document:
        """
        Convert a message to a Document object.

        Args:
            message (dict): A message in the form of a dictionary.
            channel_name (str): The name of the channel the message belongs to.

        Returns:
            Document: A Document object representing the message.
        """
        text = self._enriched_message_text(message)
        metadata = self._get_message_metadata(message, channel_name, text)
        return Document(
            page_content=text,
            metadata=metadata,
        )

    def _get_message_metadata(
        self,
        message: dict,
        channel_name: str,
        message_text: str,
    ) -> dict:
        """Create and return metadata for a given message and channel."""
        timestamp = message.get("ts", "")
        user = message.get("user", "")
        source = self._get_message_source(channel_name, user, timestamp)
        message_metadata = {
            "source": source,
            "channel": channel_name,
            "timestamp": timestamp,
            "user": user,
            "size": f"{len(message_text)}",
        }
        if self.load_auth:
            authorized_identities = self._authorized_identities_map.get(
                channel_name, []
            )
            message_metadata["authorized_identities"] = authorized_identities or []
            name = self.user_details_map.get(user, {}).get("real_name")
            message_metadata["owner"] = name or user
        return message_metadata

    def _get_message_source(self, channel_name: str, user: str, timestamp: str) -> str:
        """
        Get the message source as a string.

        Args:
            channel_name (str): The name of the channel the message belongs to.
            user (str): The user ID who sent the message.
            timestamp (str): The timestamp of the message.

        Returns:
            str: The message source.
        """
        if self.workspace_url:
            channel_id = self.channel_details_map.get(channel_name, {}).get("id")
            return (
                f"{self.workspace_url}/archives/{channel_id}"
                + f"/p{timestamp.replace('.', '')}"
            )
        else:
            return f"{channel_name} - {user} - {timestamp}"
