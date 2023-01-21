import { Box, Container, Flex, Progress, Stack, Tabs } from "@mantine/core";
import { useState } from "react";
import { Chessboard } from "react-chessboard";
import { ChessGame } from "./components/ChessGame";
import { Indicator } from "./components/indicator";

function App() {
  return (
    <Box w="100vw" h="100vh" mt="xl">
      <Container>
        <Tabs defaultValue="gallery">
          <Tabs.List>
            <Tabs.Tab value="play">Play</Tabs.Tab>
            <Tabs.Tab value="set">Set</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="play" pt="xs">
            <Flex align="stretch" mt="xl" h="100%">
              <ChessGame />
              <Indicator white={0.5} />
              <Indicator white={0.5} />
            </Flex>
          </Tabs.Panel>

          <Tabs.Panel value="set" pt="xs">
            set
          </Tabs.Panel>
        </Tabs>
      </Container>
    </Box>
  );
}

export default App;
